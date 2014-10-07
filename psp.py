from __future__ import print_function, division

import os
from shutil import copy
import subprocess
from subprocess import call, Popen, PIPE
import string

import numpy as np

pi=3.14159265359

# define options keys and default values for 4 types of options files
OPTS_DEFAULT = {'daophot': {'read_noise': 6.70, 'gain': 4.20, 'low_good': 7.00, 'high_good': 32766.50, 'fwhm': 2.50, 'sigma_th': 4.00, 'low_sharp': 0.30, 'high_sharp':1.40, 'low_round': -1.00, 'high_round':1.00, 'watch':0.00, 'r_fit': 15.00, 'r_psf': 20.00, 'var_psf': 2.00, 'sky': 0.00, 'psf_model': 5.00, 'psf_clean': 5.00, 'use_sat': 0.00, 'err_percent': 0.75, 'err_profile': 5.00},
'photo': {'a1': 3, 'a2': 4, 'a3': 5, 'a4': 6, 'a5': 7, 'a6': 8, 'a7': 10, 'a8': 11, 'a9': 12, 'aa': 13, 'ab': 14, 'ac': 15, 'inner_sky': 20., 'outer_sky': 35.},
'allstar': {'r_fit': 15.00, 'clip_exp': 6.00, 'redet_centroid': 1.00, 'clip_range': 2.50, 'watch': 0.00, 'group_max': 50.00, 'err_percent': 0.75, 'err_profile': 5.00, 'inner_sky': 20., 'outer_sky': 35.},
'allframe': {'clip_exp': 6.00, 'clip_range': 2.50, 'geo_coeff': 6.00, 'watch': 0.00, 'iter_min': 5, 'iter_max': 200, 'err_percent': 0.75, 'err_profile': 5.00, 'inner_sky': 20., 'outer_sky': 35.}}

# map option keys to daophot parameters
OPTS_KEYS = {'daophot': {'read_noise': 'READ NOISE (ADU; 1 frame)', 'gain': 'GAIN (e-/ADU; 1 frame)', 'low_good': 'LOW GOOD DATUM (in sigmas)', 'high_good': 'HIGH GOOD DATUM (in ADU)', 'fwhm': 'FWHM OF OBJECT', 'sigma_th': 'THRESHOLD (in sigmas)', 'low_sharp': 'LS (LOW SHARPNESS CUTOFF)', 'high_sharp': 'HS (HIGH SHARPNESS CUTOFF)', 'low_round': 'LR (LOW ROUNDNESS CUTOFF)', 'high_round': 'HR (HIGH ROUNDNESS CUTOFF)', 'watch': 'WATCH PROGRESS', 'r_fit': 'FITTING RADIUS', 'r_psf': 'PSF RADIUS', 'var_psf': 'VARIABLE PSF', 'sky': 'SKY ESTIMATOR', 'psf_model': 'ANALYTIC MODEL PSF', 'psf_clean': 'EXTRA PSF CLEANING PASSES', 'use_sat': 'USE SATURATED PSF STARS', 'err_percent': 'PERCENT ERROR (in %)', 'err_profile': 'PROFILE ERROR (in %)'},
'photo': {'a1': 'A1', 'a2': 'A2', 'a3': 'A3', 'a4': 'A4', 'a5': 'A5', 'a6': 'A6', 'a7': 'A7', 'a8': 'A8', 'a9': 'A9', 'aa': 'AA', 'ab': 'AB', 'ac': 'AC', 'inner_sky': 'IS', 'outer_sky': 'OS'},
'allstar': {'r_fit': 'FITTING RADIUS', 'clip_exp': 'CE (CLIPPING EXPONENT)', 'redet_centroid': 'REDETERMINE CENTROIDS', 'clip_range': 'CR (CLIPPING RANGE)', 'watch': 'WATCH PROGRESS', 'group_max': 'MAXIMUM GROUP SIZE', 'err_percent': 'PERCENT ERROR', 'err_profile': 'PROFILE ERROR', 'inner_sky': 'IS', 'outer_sky': 'OS'},
'allframe': {'clip_exp': 'CE (CLIPPING EXPONENT)', 'clip_range': 'CR (CLIPPING RANGE)', 'geo_coeff': 6.00, 'watch': 'WATCH PROGRESS', 'iter_min': 'MINIMUM ITERATIONS', 'iter_max': 'MAXIMUM ITERATIONS', 'err_percent': 'PERCENT ERROR', 'err_profile': 'PROFILE ERROR', 'inner_sky': 'IS', 'outer_sky': 'OS'}}

class Dao(object):
	def __init__(self, cmd, opts):
		"""Initialization for Dao objects"""
		
		# if which cmd not found -> go install, else
		self.cmd = cmd
		
		# options
		self.opts = opts
	
	def init_names(self, fname):
		# path to image
		self.fname = fname
		
		# root name 
		self.name = os.path.splitext(fname)[0]
		
		# working directory
		self.dr = os.path.split(fname)[0]
	
	def write_opts(self, op_type, pars={}, optname=None, general_name=False):
		"""Write options file"""
		if op_type not in OPTS_KEYS.keys():
			print("options type needs to be one of the following: {0}".format(OPTS_KEYS.keys()))
			return
		
		if general_name:
			optname = self.dr+op_type+".opt"
		
		if optname==None:
			self.optname = self.name.replace("/","").replace("..","")+op_type+".opt"
		else:
			self.optname = optname
		
		# check which parameters remain default
		dkeys = [x for x in OPTS_KEYS[op_type].keys() if x not in pars.keys()]
		
		# write options file
		f = open(self.optname, 'w')
		
		# input parameters
		for k in pars.keys():
			f.write("{0} = {1}\n".format(OPTS_KEYS[op_type][k], pars[k]))
		
		# default parameters
		for k in dkeys:
			f.write("{0} = {1}\n".format(OPTS_KEYS[op_type][k], OPTS_DEFAULT[op_type][k]))
		
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
		# check p.returncode for !=0, raise error, check daophot log, stop here
		if p.poll()==None:
			p.terminate()
		
		return stdoutdata
	
	def find_instream(self, stream, search_string):
		"""Return index in stream where search_string ends"""
		ind = string.find(stream, search_string)
		s1 = ind + len(search_string)
		
		return s1
		
class Find(Dao):
	def __init__(self, opts):
		super(Find, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
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
END_DAOPHOT""".format(self.optname, self.opts['misc']['stacknum'], self.name)
		
		# call daophot find
		stdout = self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
		search_string = "Sky mode and standard deviation = "
		s1 = self.find_instream(stdout, search_string)
		self.opts['misc']['sky_mode'] = float(stdout[s1:s1+8])

class Aper(Dao):
	def __init__(self, opts):
		super(Aper, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('photo', self.opts['photo'], general_name=True)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', self.opts['daophot'])
		
		# build call
		lines="""options
{0}

attach {1}.fits

photometry


{1}.coo
{1}.ap

exit
END_DAOPHOT""".format(self.optname, self.name)
		
		# call daophot aperture photometry
		stdout = self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
		search_string = "Estimated magnitude limit (Aperture 1): "
		s1 = self.find_instream(stdout, search_string)
		self.opts['misc']['maglim'] = float(stdout[s1:s1+4])

class PickStars(Dao):
	def __init__(self, opts):
		super(PickStars, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
		# build call
		lines="""options
{0}

attach {1}.fits

pickpsf
{1}.ap
100,{2}
{1}.lst1

exit
END_DAOPHOT""".format(self.optname, self.name, self.opts['misc']['maglim'])
		
		# call daophot find
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
class GetPSF(Dao):
	def __init__(self, opts):
		super(GetPSF, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
		# build call
		lines="""options
{0}

attach {1}.fits

psf
{1}.ap
{1}.lst1
{1}.psf

exit
END_DAOPHOT""".format(self.optname, self.name)
		
		# call daophot find
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])

# general helper functions
def cleanup_files(fname, extensions):
	"""Remove files with root fname and extensions"""
	for ext in extensions:
		silent_remove(os.path.splitext(fname)[0]+".{0}".format(ext))

def silent_remove(fname):
	"""Remove file if exists"""
	try:
		os.remove(fname)
	except OSError:
		pass

def set_indopts(fname, opts):
	"""Read gain, readnoise and saturation level for input fname and store to opts"""
	
	info = os.path.splitext(fname)[0]+".info"
	keys = ('gain', 'read_noise', 'high_good')
	
	if os.path.isfile(info)==False:
		print("Info file not found, proceeding with defaults.")
		for i, key in enumerate(keys):
			opts['daophot'][key] = OPTS_DEFAULT['daophot'][key]
	else:
		op = np.loadtxt(info)
		for i, key in enumerate(keys):
			opts['daophot'][key] = op[i]

def test():
	"""Testing new photometric pipeline"""
	
	# global options
	opts = {'daophot': {'r_psf': 20, 'r_fit': 15}, 'photo': {}, 'allstar': {}, 'allframe': {}, 'misc': {'stacknum': 1, 'counts_limit': 15000, 'number_limit': 400, 'sigma_psf': 4.5, 'sigma_all': 2.5}}
	
	# image
	fname = "test.fits"
	
	# individual options
	set_indopts(fname, opts)
	
	# psf image (masked image)
	os.system("python psf_mask.py {0} {1} {2} {3}".format(fname, opts['misc']['counts_limit'], opts['misc']['number_limit'], opts['daophot']['high_good']+1))
	psfname = os.path.splitext(fname)[0]+"_psf.fits"
	# if psf image not created, copy the original image
	if os.path.isfile(psfname)==False:
		copy(fname, psfname)
		
	# cleanup
	cleanup_files(fname, ("log", "coo"))
	cleanup_files(psfname, ("log", "coo"))
	
	# create photometry objects
	f = Find(opts)
	aper = Aper(opts)
	pick = PickStars(opts)
	psf = GetPSF(opts)
	
	# initial find, to get fwhm
	f(fname)
	aper(fname)
	
	# once initial find run, determine fwhm, update it in opts
	
	print(opts)
	
	# options for psf finding
	opts['daophot']['sigma_th']=opts['misc']['sigma_psf']
	
	f(psfname)
	aper(psfname)
	pick(psfname)
	psf(psfname)
	
	# print plog
	
	# identify psf stars that have bad pixels and remove them from list
	
	# check none left
	
	
	
	
	
	
	



