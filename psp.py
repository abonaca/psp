from __future__ import print_function, division

from math import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import astropy
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import ICRS
from astropy.coordinates.angles import Angle, Longitude
from astropy.table import Table

import scipy.stats
from scipy.optimize import leastsq
from scipy.spatial import cKDTree

import random
import time
import glob
import os
import subprocess
from subprocess import call, Popen, PIPE
from myutils import *

from collections import Counter
import string
import threading

pi=3.14159265359

class Dao(object):
	def __init__(self, fname, cmd, opts):
		"""Initialization for Dao objects"""
		
		# path to image
		self.fname = fname
		
		# root name
		self.name = fname[:fname.rfind('.')]
		
		# working directory
		sidx = self.name.rfind('/')
		if sidx == -1:
			self.dr = ""
		else:
			self.dr = self.name[:sidx]
		
		# if which cmd not found -> go install, else
		self.cmd = cmd
		
		# options
		self.opts = opts
		
		# define options keys and default values for 4 types of options files
		self.opkeys = {'daophot': {'re': 6.70, 'ga': 4.20, 'lo': 7.00, 'hi': 32766.50, 'fw': 2.50, 'th': 4.00, 'ls': 0.30, 'hs':1.40, 'lr': -1.00, 'hr':1.00, 'wa':0.00, 'fi': 15.00, 'ps': 20.00, 'va': 2.00, 'sk': 0.00, 'an': 5.00, 'ex': 5.00, 'us': 0.00, 'pe': 0.75, 'pr': 5.00},
		'photo': {'a1': 3, 'a2': 4, 'a3': 5, 'a4': 6, 'a5': 7, 'a6': 8, 'a7': 10, 'a8': 11, 'a9': 12, 'aa': 13, 'ab': 14, 'ac': 15, 'is': 20., 'os': 35.},
		'allstar': {'fi': 15.00, 'ce': 6.00, 're': 1.00, 'cr': 2.50, 'wa': 0.00, 'ma': 50.00, 'pe': 0.75, 'pr': 5.00, 'is': 20., 'os': 35.},
		'allframe': {'ce': 6.00, 'cr': 2.50, 'ge': 6.00, 'wa': 0.00, 'mi': 5, 'ma': 200, 'pe': 0.75, 'pr': 5.00, 'is': 20., 'os': 35.}}
		
	
	def write_opts(self, op_type, pars={}, optname=None, general_name=False):
		"""Write options file"""
		if op_type not in self.opkeys.keys():
			print("options type needs to be one of the following: {0}".format(self.opkeys.keys()))
			return
		
		if general_name:
			optname = self.dr+op_type+".opt"
		
		if optname==None:
			self.optname = self.name.replace("/","").replace("..","")+op_type+".opt"
		else:
			self.optname = optname
		
		# check which parameters remain default
		dkeys = [x for x in self.opkeys[op_type].keys() if x not in pars.keys()]
		
		# write options file
		f = open(self.optname, 'w')
		
		# input parameters
		for k in pars.keys():
			f.write("{0} = {1}\n".format(k, pars[k]))
		
		# default parameters
		for k in dkeys:
			f.write("{0} = {1}\n".format(k, self.opkeys[op_type][k]))
		
		f.close()
	
	def run_cmd(self, lines, args=None):
		"""Run a dao-family command"""
		# build command
		cmd = [self.cmd]
		cmd.extend(args)
		
		# run command
		p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
		stdoutdata, stderrdata=p.communicate(lines)
		
		# save log files
		f = open(self.name+".log", 'a')
		f.write(stdoutdata)
		f.close()
		#print("out: ", stdoutdata)
		#print("err: ", stderrdata)
		
		# close process
		if p.poll==None:
			p.terminate()
		
		
class Find(Dao):
	def __init__(self, img, opts):
		super(Find, self).__init__(img, 'daophot', opts)
		
	def doit(self):
		"""Call daophot and run find command"""
		
		# build call
		lines="""options
{0}

attach {2}.fits

find
{1},1
{2}.coo
y

exit
END_DAOPHOT""".format(self.optname, self.opts['stacknum'], self.name)
		
		# call daophot find
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])

def silent_remove(fname):
	"""Remove file if exists"""
	try:
		os.remove(fname)
	except OSError:
		pass
	
def test():
	fname = "test.fits"
	cmd = "daophot"
	opts = {'rpsf': 20, 'rfit': 15, 'stacknum': 1}
	f = Find(fname, opts)
	print(f.name, f.dr)
	
	# cleanup
	for ext in ("log", "coo"):
		silent_remove(f.name+".{0}".format(ext))
	
	f.write_opts('daophot', general_name=True)
	f.write_opts('daophot', pars={'ps':f.opts['rpsf'], 'fi':f.opts['rfit']})
	f.doit()

	
	



