#!/usr/bin/env python2

import a3

import os
import logging
import h5py
import curses

logging.basicConfig(filename='a3.log',level=logging.INFO)

class A3CL():
    def __init__(self):
        self.win = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.win.keypad(1)
        self.menu = Menu()

    def __enter__(self):
        return self

    def __call__(self):
        self.menu.refresh(self.x)
        while True:
            c = self.win.getch()
            if c == ord('q'):
                break
            
    @property
    def y(self):
        return self.win.getmaxyx()[0]
    
    @property
    def x(self):
        return self.win.getmaxyx()[1]

    def __exit__(self, a,b,c):
        curses.nocbreak()
        self.win.keypad(0)
        curses.echo()
        curses.endwin()

class Menu():
    def __init__(self):
        self.win = curses.newpad(1,1024)
        self.win.addstr(0,0,"^N(ew)  ^O(pen)") 
        self.win.chgat(0,0,-1,curses.A_REVERSE)
        self.x = 0
        
    def refresh(self,width):
        self.win.refresh(2,2, 0,0, 1,20)


with A3CL() as session:
    session()
