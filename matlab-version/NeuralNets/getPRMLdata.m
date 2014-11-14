function PRMLdata = getPRMLdata(typeofdata)

switch typeofdata
case 'curve'
% y = sin(2πx) + N(0,0.3)
PRMLdata  = [0.000000, 0.349486 ; ...
             0.111111, 0.830839 ; ...
             0.222222, 1.007332 ; ...
             0.333333, 0.971507 ; ...
             0.444444, 0.133066 ; ...
             0.555556, 0.166823 ; ...
             0.666667, -0.848307; ...
             0.777778, -0.445686; ...
             0.888889, -0.563567; ...
             1.000000, 0.261502 ];

case 'gaussians'
% The classification data contains 200 data, sampled from a 3-component Gaussian mixture in 2D. 
% This data was generated using the gmmsamp function from Netlab. The corresponding Gaussian 
% mixture model had the parameters:
% 
% mix.priors = [0.5 0.25 0.25];
% mix.centres = [0 -0.1; 1 1; 1 -1];
% mix.covars(:,:,1) = [0.625 -0.2165; -0.2165 0.875];
% mix.covars(:,:,2) = [0.2241 -0.1368; -0.1368 0.9759];
% mix.covars(:,:,3) = [0.2375 0.1516; 0.1516 0.4125];
% 
% The first component represent class 1 (blue circles, o, in the left panel of Figure A.7),
% the other components class 0 (red crosses, ×). The file has 200 rows of 3 columns, the first
% two columns giving datum position, the last column containing the label (0/1).

PRMLdata =[1.208985 0.421448 0.000000  ;...
           0.504542 -0.285730 1.000000 ;...
           0.630568 1.054712 0.000000  ;...
           1.056364 0.601873 0.000000  ;...
           1.095326 -1.447579 1.000000 ;...
           -0.210165 0.000284 1.000000 ;...
           -0.367151 -1.255189 1.000000;...
           0.868013 -1.063465 0.000000 ;...
           1.704441 -0.644833 0.000000 ;...
           0.565619 -1.637858 1.000000 ;...
           0.598389 -1.477808 0.000000 ;...
           0.580927 -0.783898 1.000000 ;...
           1.183283 -1.797936 0.000000 ;...
           0.331843 -1.869486 0.000000 ;...
           -0.051195 0.989475 1.000000 ;...
           2.427090 0.173557 0.000000  ;...
           1.603778 -0.030691 1.000000 ;...
           1.286206 -1.079916 0.000000 ;...
           -1.243951 1.005355 1.000000 ;...
           1.181748 1.523744 0.000000  ;...
           0.896222 1.899568 0.000000  ;...
           -0.366207 -0.664987 1.000000;...
           -0.078800 1.007368 1.000000 ;...
           -1.351435 1.766786 1.000000 ;...
           -0.220423 -0.442405 1.000000;...
           0.836253 -1.927526 0.000000 ;...
           0.039899 -1.435842 0.000000 ;...
           0.256755 0.946722 0.000000  ;...
           0.974836 -0.944967 0.000000 ;...
           0.705256 -2.618644 0.000000 ;...
           0.738188 -1.666242 0.000000 ;...
           1.245931 -2.200826 0.000000 ;...
           0.297604 0.159463 1.000000  ;...
           -2.210680 1.195815 1.000000 ;...
           -0.872624 -0.131252 1.000000;...
           1.112762 -0.653777 0.000000 ;...
           1.123989 -1.347470 0.000000 ;...
           0.750833 0.811870 0.000000  ;...
           -0.183497 1.416116 1.000000 ;...
           0.287582 -1.342512 0.000000 ;...
           1.092719 1.380559 0.000000  ;...
           0.719502 1.594624 0.000000  ;...
           -1.016254 0.651607 1.000000 ;...
           0.379677 2.802498 0.000000  ;...
           0.150675 0.474679 1.000000  ;...
           -0.116477 0.437483 1.000000 ;...
           1.122528 0.698541 0.000000  ;...
           0.953551 1.088368 0.000000  ;...
           -0.000228 0.347187 1.000000 ;...
           0.505024 0.455407 1.000000  ;...
           0.113753 0.559572 1.000000  ;...
           -0.677993 0.322716 1.000000 ;...
           1.114811 -0.735813 0.000000 ;...
           0.344114 -1.770137 0.000000 ;...
           0.684242 -0.636027 1.000000 ;...
           -0.684629 -0.300568 1.000000;...
           -0.362677 -0.669101 1.000000;...
           0.604984 -1.558581 0.000000 ;...
           0.514202 -0.225827 0.000000 ;...
           0.227014 -1.579346 1.000000 ;...
           1.044068 -1.491114 0.000000 ;...
           0.314855 -2.535762 1.000000 ;...
           1.187904 -1.367278 0.000000 ;...
           0.517132 1.375811 0.000000  ;...
           1.244285 -0.764164 0.000000 ;...
           -0.831841 1.728708 1.000000 ;...
           1.719616 -2.491282 1.000000 ;...
           0.594216 1.137571 1.000000  ;...
           0.939919 -0.474988 0.000000 ;...
           -0.918736 -0.748474 1.000000;...
           0.913760 -1.194336 0.000000 ;...
           0.893221 -1.569459 0.000000 ;...
           0.653152 0.510498 0.000000  ;...
           0.766890 -1.577565 0.000000 ;...
           0.868315 -1.966740 1.000000 ;...
           0.874218 0.514959 1.000000  ;...
           -0.559543 1.749552 1.000000 ;...
           1.526669 -1.797734 1.000000 ;...
           1.843439 -0.363161 0.000000 ;...
           1.163746 2.062245 0.000000  ;...
           0.565749 -2.432301 1.000000 ;...
           1.016715 2.878822 0.000000  ;...
           1.433979 -1.944960 1.000000 ;...
           -0.510225 0.295742 1.000000 ;...
           -0.385261 0.278145 1.000000 ;...
           1.042889 -0.564351 0.000000 ;...
           -0.607265 1.885851 1.000000 ;...
           -0.355286 -1.813131 1.000000;...
           -0.790644 -0.790761 1.000000;...
           1.372382 0.879619 0.000000  ;...
           1.133019 -0.300956 0.000000 ;...
           1.395009 -1.006842 0.000000 ;...
           0.887843 0.222319 1.000000  ;...
           1.484690 0.095074 0.000000  ;...
           1.268061 1.832532 0.000000  ;...
           0.124568 0.910824 1.000000  ;...
           1.061504 -0.768175 1.000000 ;...
           0.298551 2.573175 0.000000  ;...
           0.241114 -0.613155 0.000000 ;...
           -0.423781 -1.524901 1.000000;...
           0.528691 -0.939526 0.000000 ;...
           1.601252 1.791658 0.000000  ;...
           0.793609 0.812783 1.000000  ;...
           0.327097 0.326998 0.000000  ;...
           1.131868 -0.985696 1.000000 ;...
           1.273154 1.656441 0.000000  ;...
           -0.816691 0.961580 1.000000 ;...
           0.669064 1.162614 0.000000  ;...
           -0.453759 -1.146883 1.000000;...
           2.055105 0.025811 0.000000  ;...
           0.463119 -0.813294 1.000000 ;...
           0.802392 -0.140807 1.000000 ;...
           -0.730255 -0.145175 1.000000;...
           0.569256 0.567628 1.000000  ;...
           0.486947 1.130519 0.000000  ;...
           1.793588 -1.426926 0.000000 ;...
           1.178831 -0.581314 1.000000 ;...
           0.480055 1.257981 0.000000  ;...
           0.683732 0.190071 1.000000  ;...
           -0.119082 -0.004020 1.000000;...
           -1.251554 -0.176027 1.000000;...
           1.094741 -1.099305 0.000000 ;...
           -0.238250 -1.277484 1.000000;...
           -0.661556 1.327722 1.000000 ;...
           1.442837 1.241720 0.000000  ;...
           1.202320 0.489702 0.000000  ;...
           0.932890 0.296430 0.000000  ;...
           0.665568 -1.314006 0.000000 ;...
           -0.058993 1.322294 1.000000 ;...
           0.209525 -1.006357 0.000000 ;...
           1.023340 0.219375 0.000000  ;...
           1.324444 0.446567 1.000000  ;...
           1.453910 -1.151325 0.000000 ;...
           0.616303 0.974796 0.000000  ;...
           1.492010 -0.885984 0.000000 ;...
           1.738658 0.686807 1.000000  ;...
           0.900582 -0.280724 0.000000 ;...
           0.961914 -0.053991 1.000000 ;...
           1.819706 -0.953273 1.000000 ;...
           1.581289 -0.340552 0.000000 ;...
           0.520837 -0.680639 1.000000 ;...
           1.433771 -0.914798 0.000000 ;...
           0.611594 -1.691685 0.000000 ;...
           1.591513 -0.978986 1.000000 ;...
           1.282094 0.113769 0.000000  ;...
           0.985715 0.275551 0.000000  ;...
           -1.805143 2.628696 1.000000 ;...
           1.473100 -0.241372 0.000000 ;...
           -0.242212 -1.040151 1.000000;...
           1.175525 -1.662026 0.000000 ;...
           0.696040 0.154387 0.000000  ;...
           1.457713 1.608681 0.000000  ;...
           0.883215 1.330538 0.000000  ;...
           -0.681209 0.622394 1.000000 ;...
           -0.355082 0.432941 1.000000 ;...
           0.633011 -1.194431 0.000000 ;...
           0.782723 1.060008 1.000000  ;...
           0.670180 -0.766999 1.000000 ;...
           -0.047154 0.698693 1.000000 ;...
           0.287385 -1.097756 0.000000 ;...
           0.069561 1.632585 1.000000  ;...
           1.013230 1.111551 0.000000  ;...
           0.639065 -0.697237 0.000000 ;...
           1.174621 2.240022 1.000000  ;...
           1.322020 0.040277 1.000000  ;...
           0.019127 0.105667 1.000000  ;...
           0.584584 1.101914 0.000000  ;...
           1.157265 -0.665947 0.000000 ;...
           1.565230 -0.840790 0.000000 ;...
           1.759315 0.963703 1.000000  ;...
           1.687068 -1.086466 0.000000 ;...
           0.578314 -0.340961 1.000000 ;...
           0.118925 -1.487694 1.000000 ;...
           0.471201 0.330872 1.000000  ;...
           -0.268209 -0.353477 0.000000;...
           1.625390 -1.718798 0.000000 ;...
           1.117791 2.752549 0.000000  ;...
           -0.194552 -0.752687 1.000000;...
           0.769548 -2.066152 0.000000 ;...
           0.186062 0.022072 1.000000  ;...
           1.771337 -0.393550 0.000000 ;...
           -1.300597 0.962803 1.000000 ;...
           0.708730 -1.013371 0.000000 ;...
           -0.624235 -0.892995 1.000000;...
           0.377055 -1.296098 0.000000 ;...
           0.804404 -0.856253 1.000000 ;...
           1.359887 -0.974291 0.000000 ;...
           -0.115505 0.228439 1.000000 ;...
           0.913645 -0.344936 1.000000 ;...
           0.318875 -0.886290 1.000000 ;...
           0.822157 0.102548 0.000000  ;...
           -0.281208 1.302572 1.000000 ;...
           0.044639 -1.107980 1.000000 ;...
           -0.029205 -2.033973 0.000000;...
           0.879914 -2.000582 1.000000 ;...
           0.601936 -0.503923 0.000000 ;...
           -0.490114 -0.841122 1.000000;...
           1.847075 2.362322 0.000000  ;...
           -0.279703 0.753196 1.000000 ;...
           1.953357 -0.746632 0.000000 ];
           
otherwise
    error('must be either curve or gaussians')
end