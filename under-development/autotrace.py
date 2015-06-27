#!/usr/bin/env python2

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
import warnings
import json
from glob import glob
import fnmatch
import Image

# These evironment flags set up GPU training
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu'

import numpy as np
import h5py
import theano
import theano.tensor as T
import lasagne


warnings.filterwarnings('ignore','.*topo.*')
warnings.filterwarnings('ignore','.*Glorot.*')

_version = (0,1,0)

class ROI(object):
    """ Region of Interest for a set of images
    
    Attributes:
        shape (tuple of numeric): the height and width of the ROI
        offset (tuple of numeric): the lower bounds of the ROI
        extent (tuple of numeric): the upper bounds of the ROI
            offset[dim] + shape[dim] should always == extent[dim]
        orthodox (tuple of bool): whether the ROI is indexed "normally"
            I.e. if the ROI is measured from the top/left
            If measured from the bottom-left: (False, True)
        slice (tuple of slice): can be used to slice into a 2d matrix
            >>> np.identity(5)[ROI(2,3,1,4).slice]
            array([[ 0., 1., 0.]])
        
    """
    def __init__(self,*args,**kwargs):
        """
        Multiple possible ways of declaring an ROI are supported.
        The first way is by specifying the bounds as positional args
        Args:
            top (numeric): the top of the region of interest
            bottom (numeric): the bottom of the region of interest
            left (numeric): the left edge of the region of interest
            right (numeric): the right edge of the region of interest
        Example:
            >>> ROI(1,2,3,4)
            ROI(1.0, 2.0, 3.0, 4.0)

        The second way is by specifying a single iterable object
        Example:
            >>> ROI(1,2,3,4) == ROI([1,2,3,4])
            True

        Regardless of the constructor format used, the order should
            always be: top, bottom, left, right
        This allows for symantic interpretation of the arguments.
            ROI is smart enough to deal with indexing from other edges
        Example:
            >>> ROI(2,1,4,3).slice
            (slice(1.0, 2.0, None), slice(3.0, 4.0, None))
            >>> ROI(2,1,4,3).top
            2.0
        """
        if len(args) == 4:
            roi = (args[0],args[1],args[2],args[3])
        elif len(args) == 1:
            roi = args [0]
        (top, bottom, left, right) = [float(x) for x in roi]
        self.orthodox = (top<bottom, left<right)
        self.shape  = (abs(top-bottom), abs(left-right))
        self.offset = (min(top,bottom), min(left,right))
        self.extent = (max(top,bottom), max(left,right))
        self.slice = (slice(self.offset[0],self.extent[0]),
            slice(self.offset[1],self.extent[1]))

    @property
    def top(self): 
        """Convenience property for the top of the ROI
            For an orthodox ROI, this is the same as offset[0]
            For an ROI unorthodox in the Y dimension, this is extent[0]
        """
        return self.offset[0] if self.orthodox[0] else self.extent[0]
    @property
    def bottom(self): 
        """Convenience property for the bottom of the ROI
            For an orthodox ROI, this is the same as extent[0]
            For an ROI unorthodox in the Y dimension, this is offset[0]
        """
        return self.extent[0] if self.orthodox[0] else self.offset[0]
    @property
    def left(self): 
        """Convenience property for the left of the ROI
            For an orthodox ROI, this is the same as offset[1]
            For an ROI unorthodox in the X dimension, this is extent[1]
        """
        return self.offset[1] if self.orthodox[1] else self.extent[1]
    @property
    def right(self): 
        """Convenience property for the right of the ROI
            For an orthodox ROI, this is the same as extent[1]
            For an ROI unorthodox in the X dimension, this is offset[1]
        """
        return self.extent[1] if self.orthodox[1] else self.offset[1]
    @property
    def height(self): 
        """Convenience property for the height of the ROI 
            This is the same as shape[0]
        """
        return self.shape[0]
    @property
    def width(self): 
        """Convenience property for the width of the ROI
            This is the same as shape[1]
        """
        return self.shape[1]

    def __repr__(self):
        return 'ROI(%s, %s, %s, %s)' % tuple(self)
    
    def __eq__(self,other):
        return repr(self) == repr(other)

    def __iter__(self):
        """Iterate over ROI bounds

        Yields:
            numeric: top, bottom, left, right (strictly ordered)
        """
        return (x for x in (self.top,self.bottom,self.left,self.right))

    def domain(self,N):
        """Returns a numpy array of N equally-spaced x values in the ROI
        
        Args:
            N (integer): number of points to create

        Returns:
            numpy array: N evenly-spaced points, from offset[1] to 
                extent[1] (includes neither offset[1] nor extent[1])
                The dtype should be float32

        Example:
            >>> ROI(x,y,10,20).domain(3)
            array([12.5,15.,17.5])
        """
        step = self.shape[1] / (N + 1)
        return np.arange(self.offset[1] + step, self.extent[1], step)
    
    def json(self):
        """json stringify the ROI"""
        j = {
            'srcY': self.offset[0],
            'destY': self.extent[0],
            'srcX': self.offset[1],
            'destX': self.extent[1],
        }
        return json.dumps(j)
    

    def scale(self,factor):
        """Create a scaled version of the current ROI.
        
        Args:
            factor (numeric): the factor by which to scale. 
        
        Returns:
            ROI: the scaled ROI

        Example:
            >>> ROI(1,2,3,4).scale(2.5)
            ROI(2.5, 5.0, 7.5, 10.0)
        """
        return ROI(np.array(tuple(self))*factor)
        

class Autotracer(object):
    """Automatically traces tongues in Ultracound images.
    
    Attributes (all read-only):
        roi (ROI): Where the ultrasound images the data represent.
        X_train (tensor of float32): the training dataset images.
            each element is 1 pixel, scaled from 0 (black) to 1 (white).
            If roi.shape is (y,x) then X_train.shape must be (N,1,y,x).
        y_train (tensor of float32): the training dataset traces.
            Elements represent points on the tongue relative to the roi.
            0 represents that the point lies on roi.offset[0], while
            1 represents that the point lies on roi.extent[0].
            For traces with M points, y_train.shape should be (N,M,1,1).
        X_valid (tensor of float32): the validation dataset images
            each element is 1 pixel, scaled from 0 (black) to 1 (white).
            if roi.shape is (y,x) then X_test.shape should be (N,1,y,x).
        y_valid (tensor of float32): the validation dataset traces.
            Elements represent points on the tongue relative to the roi.
            0 represents that the point lies on roi.offset[0], while
            1 represents that the point lies on roi.extent[0].
            For traces with M points, y_valid.shape should be (N,M,1,1).
        layer_in (lasagne.layers.input.InputLayer): input layer
        layer_out (lasagne.layers.dense.DesnseLayer): output layer
    """
    def __init__(self,train,test,roi):
        """
        Currently, saving and loading autotracers is not supported, 
        so only one constructor syntax is supported. 

        Args:
            train (string): the location of a hdf5 dataset for training
                this gets loaded as X_train and y_train
            test (string): the location of a hdf5 dataset for validation
                this gets loaded as X_valid and y_valid
            roi (ROI): the location of the data within an image
        """
        self.loadHDF5(train,test)
        self.roi = ROI(roi)
        self.__init_model()

    def loadHDF5(self,train,test):
        """Load a test and training dataset from hdf5 databases

        Args:
            train (string): the location of a hdf5 dataset for training
                this gets loaded as X_train and y_train
            test (string): the location of a hdf5 dataset for validation
                this gets loaded as X_valid and y_valid
        """
        print('loadHDF5(%s,%s)'%(train,test))
        with h5py.File(train,'r') as h:
            self.X_train = np.array(h['image'])
            self.y_train = np.array(h['trace'])
        with h5py.File(test,'r') as h:
            self.X_valid = np.array(h['image'])
            self.y_valid = np.array(h['trace'])

    def __init_layers(self,layer_size):
        """Create the architecture of the MLP
        
        The achitecture is currently hard-coded.
        Architecture:
            image -> ReLU w/ dropout (x3) -> trace
        Args:
            layer_size (integer): the size of each layer
                Currently, all the layers have the same number of units
                (except for the input and output layers).
        """
        self.layer_in = lasagne.layers.InputLayer(
            shape = (None,) + self.X_train.shape[1:])
        l_hidden1 = lasagne.layers.DenseLayer(
            self.layer_in,
            num_units = layer_size,
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())
        l_hidden1_d = lasagne.layers.DropoutLayer(l_hidden1, p=.5)
        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1_d,
            num_units = layer_size,
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())
        l_hidden2_d = lasagne.layers.DropoutLayer(l_hidden2, p=.5)
        l_hidden3 = lasagne.layers.DenseLayer(
            l_hidden2_d,
            num_units = layer_size,
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())
        l_hidden3_d = lasagne.layers.DropoutLayer(l_hidden3, p=.5)
        self.layer_out = lasagne.layers.DenseLayer(
            l_hidden3_d,
            num_units = self.y_train.shape[1],
            nonlinearity = lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform())

    def __init_model(self,layer_size=2048):
        """Initializes the model
        
        For the most part, this consists of setting up some bookkeeping
        for theano and lasagne, and compiling the theano functions
        Args:
            layer_size (integer): the size of each layer in the MLP
            gets passed directly to self.__init_layers
        """
        print('initializing model')
        self.__init_layers(layer_size)

        # These are theano/lasagne symbolic variable declarationss, 
        # representing... the target vector(traces)
        target_vector = T.fmatrix('y')
        # the objective function (Mean-Sq Error)
        objective = lasagne.objectives.Objective( 
            self.layer_out,
            loss_function = lasagne.objectives.mse)
        # the loss (diff in objective) for training
        stochastic_loss = objective.get_loss( 
            target = target_vector)
        # the loss for validation
        deterministic_loss = objective.get_loss( 
            target = target_vector,
            deterministic = True)
        # the network parameters (i.e. weights)
        all_params = lasagne.layers.get_all_params( 
            self.layer_out)
        # how to update the weights
        updates = lasagne.updates.nesterov_momentum( 
            loss_or_grads = stochastic_loss, 
            params = all_params,
            learning_rate = 0.1,
            momentum = 0.9)
        
        # The theano functions for training, testing, and tracing. 
        #   These get method-level wrappers below
        print('compiling theano functions')
        self._train_fn = theano.function(
            inputs  = [self.layer_in.input_var,target_vector],
            outputs = [stochastic_loss],
            updates = updates)
        self._valid_fn = theano.function(
            inputs  = [self.layer_in.input_var, target_vector],
            outputs = [deterministic_loss,
                lasagne.layers.get_output(self.layer_out)])
        self._trace_fn = theano.function(
            inputs  = [self.layer_in.input_var],
            outputs = [lasagne.layers.get_output(self.layer_out)
                * self.roi.shape[0] + self.roi.offset[0]])
    
    def train_batch(self, X, y):
        """Train on a minibatch 
        
        Wrapper for _train_fn()
        
        Args:
            X (tensor of float32): Minibatch from the training images
            y (tensor of float32): The corresponding traces
        """
        return self._train_fn(X,y)

    def valid_batch(self, X, y):
        """Validates the network on a (mini)batch

        Wrapper for _valid_fn()

        Args:
            X (tensor of float32): Minibatch from the validation images
            y (tensor of float32): The corresponding traces
        """
        return self._valid_fn(X,y)

    def trace(self, X, jfile=None,names=None,project_id=None,subject_id=None):
        """Trace a batch of images using the MLP

        Can be used programmatically to get a numpy array of traces, 
        or a json file for use with the APIL webapp.
        Args:
            X (tensor of float32): image to be traced
                should be properly scaled to [0,1] and the roi.
            jfile (string, optional): location to save json traces
                If falsey, then no json trace is created
                The rest of the args are required if jfile is truthy
            names (list of str, semi-optional): filenames for each trace
                Used to associate traces in json with files
            project_id (json-ible object): the project id
                This is purely user-defined. How you identify projects.
                Suggestions include strings or numbers
            subject_id (json-ible object): the subject id
                This is also user-defined. How you identify subjects.
                Suggestions again include strings and numbers
        Returns:
            numpy.array of float32: traces for each image
                The traces will be scaled up to the scale of the image,
                rather than on the scale required for input.
        """
        t, = self._trace_fn(X)
        if jfile:
            domain = self.roi.domain(t.shape[1])
            js = { 'roi'     : self.roi.json(),
                'tracer-id'  : 'autotrace_%d.%d.%d'%_version,
                'project-id' : project_id,
                'subject-id' : subject_id}
            js['trace-data'] = {names[i]: [{'x': domain[j], 'y': float(t[i,j])}
                for j in range(len(domain)) if 
                float(t[i,j]) != self.roi.offset[1]] for i in range(len(t))}
            with open(jfile,'w') as f:
                json.dump(js,f)
        return t

    def train(self,num_epochs=2500,batch_size=512):
        """Train the MLP using minibatches

        Args:
            num_epochs (int): Number of times to run through the 
                training set during each epoch.
            batch_size (int): Number of images to calculate updates on
        """
        print('Training')
        for epoch_num in range(num_epochs):
            num_batches_train = int(np.ceil(len(self.X_train) / batch_size))
            train_losses = []
            for batch_num in range(num_batches_train):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num +1))
                X_batch = self.X_train[batch_slice]
                y_batch = self.y_train[batch_slice,:,0,0]
                loss, = self.train_batch(X_batch, y_batch)
                train_losses.append(loss)
            train_loss = np.mean(train_losses)
            num_batches_valid = int(np.ceil(len(self.X_valid) / batch_size))
            valid_losses = []
            list_of_traces_batch = []
            for batch_num in range(num_batches_valid):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num + 1))
                X_batch = self.X_valid[batch_slice]
                y_batch = self.y_valid[batch_slice,:,0,0]
                loss, traces_batch = self.valid_batch(X_batch, y_batch)
                valid_losses.append(loss)
                list_of_traces_batch.append(traces_batch)
            valid_loss = np.mean(valid_losses)
            print('Epoch: %d, train_loss=%f, valid_loss=%f'
                    % (epoch_num+1, train_loss, valid_loss))    


def get_from_files(d,path,roi,scale=1,n_points=32,buff=512,blacklist=[]):
    """Create an hdf5 dataset from a folder of images and traces

    Tries to match names of traces with names of images.  
    Args:
        d (str): The path of a folder.
            The folder is recursively searched.
        path (str): Where to save the dataset
            Any existing file will be overwritten without warning
        roi (ROI): The partof each image to extract.
        scale (numeric, optional):
            A factor by which to scale the images.
            Defaults to 1 (no scaling). A better setting might be 0.1
        n_points (int, optional): The number of points in each trace
            Defaults to 32
        buff (int, optional): Number of images to buffer before writing
            Defaults to 512
        blacklist (container): Set of image filenames to ignore
            This is particularly useful for making disjoint training / 
                testing datasets
            Defaults to the empty list (i.e. nothing excluded)
    """
    images = []
    traces = []
    names = []
    roi = ROI(roi)
    roi_s = roi.scale(scale)
    if os.path.exists(path):
        os.remove(path)
    hp = h5py.File(path,'w')
    hp.create_dataset('image',
        (0,1) + roi_s.shape,
        maxshape = (None,1) + roi_s.shape,
        chunks = (buff,1) + roi_s.shape)
    hp.create_dataset('trace',
        (0,n_points,1,1),
        maxshape = (None,n_points,1,1),
        chunks = (buff,n_points,1,1))
    hp.create_dataset('name',
        (0,),
        maxshape = (None,),
        chunks = (buff,),
        dtype=h5py.special_dtype(vlen=unicode))
    # traverse d 
    for root,__,filenames in os.walk(d):
        # look for hand-traced traces
        for filename in fnmatch.filter(filenames,'*.ghp.traced.txt'):
            # because it matched the above fnmatch, we can assume it 
            # ends with '.ghp.traced.txt' and remove that ending.
            # the rest is our target
            base = filename[:-len('.ghp.traced.txt')]
            # look for our target
            f = None
            if os.path.isfile(os.path.join(root,base)):
                f = os.path.join(root,base)
            else:
                g = glob(os.path.join(root,'..','[sS]ubject*','IMAGES',base))
                if g:
                    f = g[0]
            # if we found it, then put it and our trace in the list
            if f:
                if os.path.basename(f) not in blacklist:
                    image = image_from_file(f,roi,scale)
                    trace = trace_from_file(os.path.join(root,filename),
                        roi,n_points)
                    try:
                        if image.any() and trace.any():
                            images.append(image)
                            traces.append(trace)
                            names.append( os.path.basename(f) )
                    except AttributeError:
                        print(image, trace)
                        raise
                else:
                    print("excluding file: %s" % (os.path.basename(f)))
            if len(images) >= buff:
                s = hp['image'].shape[0]
                images_add = np.array(images[:buff],dtype='float32')
                traces_add = np.array(traces[:buff],dtype='float32')
                hp['image'].resize(s+buff,0)
                hp['image'][s:] = images_add
                hp['trace'].resize(s+buff,0)
                hp['trace'][s:] = traces_add 
                hp['name'].resize(s+buff,0)
                hp['name'][s:] = names[:buff] 
                images = images[buff:]
                traces = traces[buff:]
                names = names[buff:]
                print(hp['image'].shape, hp['trace'].shape, hp['name'].shape)
    print(path, hp['image'].shape,hp['trace'].shape)
    hp.close()

                
def image_from_file(f,roi,scale=.01):
    """Extract a porperly scaled section of an image

    Args:
        f (str): The path to an image
        roi (ROI): The part of the image to extract
        scale
    """
    roi = ROI(roi)
    roi_scale = roi.scale(scale)
    img = Image.open(f)
    img = img.convert('L')
    img.thumbnail((img.size[0] * scale, img.size[1] * scale))
    img = np.array(img,dtype='float32')
    img = img / 255
    img = np.array(img[roi_scale.slice],dtype='float32')
    img = img.reshape(1,img.shape[0],img.shape[1])
    return img


def trace_from_file(fname,roi,n_points):
    """Extract a trace from a trace file

    Uses a linear interpolation of the trace to extract evenly-spaced points
    Args:
        fname (str): The path to a trace file.
        roi (ROI): The space accross which to evenly space the points
        n_points (int): The nuber of points to extract
    """
    roi = ROI(roi)
    gold_xs = []
    gold_ys = []
    with open(fname) as f:
        for l in f:
            l = l.split()
            if int(l[0]) > 0:
                gold_xs.append(float(l[1]))
                gold_ys.append(float(l[2]))
    gold_xs = np.array(gold_xs,dtype='float32')
    gold_ys = np.array(gold_ys,dtype='float32')
    if len(gold_xs) > 0: 
        trace = np.interp(roi.domain(n_points),gold_xs,gold_ys,left=0,right=0)
        trace = trace.reshape((n_points,1,1))
        trace[trace==0] = roi.offset[0]
        trace = (trace - roi.offset[0]) / (roi.height)
    else:
        return np.array(0)
    if trace.sum() > 0 :
        return trace
    else: 
        return np.array(0)


if __name__ == '__main__':
    roi = ROI(140.,320.,250.,580.)
    if not os.path.isfile('test.hdf5'):
        get_from_files('test_data','test.hdf5',roi,buff=10,scale=.1)
    if not os.path.isfile('train.hdf5'):
        with h5py.File('test.hdf5','r') as h:
            get_from_files('apil-data/Interspeech2014_exp/',
                'train.hdf5',roi,scale=.1,blacklist=set(h['name']))
    a = Autotracer('train.hdf5','test.hdf5',roi)
    a.train()
    with h5py.File('test.hdf5','r') as h:
        a.trace(a.X_valid,'traces.json',h['name'],'autotest','042')
