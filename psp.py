from __future__ import print_function, division

import os
from shutil import copy, move
import subprocess
from subprocess import call, Popen, PIPE
import glob
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
		"""Initialize filenames"""
		# path to image
		self.fname = fname
		
		# name without extension
		self.name = os.path.splitext(fname)[0]
		
		# only file name
		self.ofname = os.path.splitext(os.path.split(fname)[1])[0]
		
		# working directory
		self.dr = os.path.split(fname)[0]
	
	def write_opts(self, op_type, pars={}, optname=None, general_name=False):
		"""Write options file"""
		if op_type not in OPTS_KEYS.keys():
			print("options type needs to be one of the following: {0}".format(OPTS_KEYS.keys()))
			return
			
		if general_name:
			optname = op_type+".opt"
			
		#if general_name & (self.cmd=='allstar'):
			#optname = self.dr+'/'+op_type+".opt"
				
		if optname==None:
			self.optname = self.ofname+op_type+".opt"
		else:
			self.optname = optname
		

		# check which parameters remain default
		dkeys = [x for x in OPTS_KEYS[op_type].keys() if x not in pars.keys()]
		
		# write options file
		f = open(self.dr+'/'+self.optname, 'w')
		
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
		#cwd = None
		#if (self.cmd == 'allstar') & (len(self.dr)>0):
		cwd = self.dr
		
		# run command
		p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
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
		s0 = string.find(stream, search_string)
		s1 = s0 + len(search_string)
		
		return (s0, s1)
	
	def rm_duplicates(self, seq):
		""" Not order preserving    """
		keys = {}
		for e in seq:
			keys[e] = 1
		return keys.keys()
		
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
END_DAOPHOT""".format(self.optname, self.opts['misc']['stacknum'], self.ofname)
		
		# call daophot find
		stdout = self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
		search_string = "Sky mode and standard deviation = "
		s1 = self.find_instream(stdout, search_string)[1]
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
END_DAOPHOT""".format(self.optname, self.ofname)
		
		# call daophot aperture photometry
		stdout = self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
		search_string = "Estimated magnitude limit (Aperture 1): "
		s1 = self.find_instream(stdout, search_string)[1]
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
{1}.lst

exit
END_DAOPHOT""".format(self.optname, self.ofname, self.opts['misc']['maglim'])
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
class GetPSF(Dao):
	def __init__(self, opts):
		super(GetPSF, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
		silent_remove(self.name+".nei")
		
		# build call
		lines="""options
{0}

attach {1}.fits

psf
{1}.ap
{1}.lst
{1}.psf

exit
END_DAOPHOT""".format(self.optname, self.ofname)
		
		# call daophot cmd
		stdout = self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
		# get a list of bad stars
		search_string = " has a bad pixel:"
		bad = []
		snew = 0
		s0, s1 = self.find_instream(stdout, search_string)
		while s0>0:
			s0 += snew
			bad.append(stdout[s0-7:s0])
			snew += s1
			s0, s1 = self.find_instream(stdout[snew:], search_string)
		
		bad_list=self.rm_duplicates(bad)
		
		# get a list of bad psf stars
		if not bad_list:
			search_string = "Profile errors:"
			s0 = self.find_instream(stdout, search_string)[1]
			search_string = "Computed"
			s1 = self.find_instream(stdout, search_string)[0]
			
			profile_errors = stdout[s0:s1].split()
			sat = ["%7s"%(profile_errors[i - 1]) for i, x in enumerate(profile_errors) if x == "saturated"]
			var = ["%7s"%(profile_errors[i - 2]) for i, x in enumerate(profile_errors) if x == "*" or x == "?"]
			bad_list.extend(sat + var)
		
		return bad_list

class GroupStars(Dao):
	def __init__(self, opts):
		super(GroupStars, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
		# build call
		lines = """options
{0}

attach {1}.fits

group
{1}.ap
{1}.psf
0.1
{1}.grp

exit
END_DAOPHOT""".format(self.optname, self.ofname)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])
		
		out = open(self.name+'.grpt', 'w')
		f = open(self.name+'.grp', 'r')
		for line in f:
			if '-99.999' not in line:
				out.write(line)
		f.close()
		out.close()
		move(self.name+'.grpt', self.name+'.grp')

class NStar(Dao):
	def __init__(self, opts):
		super(NStar, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
		# build call
		lines = """options
{0}

attach {1}.fits

nstar
{1}.psf
{1}.grp
{1}.nst

exit
END_DAOPHOT""".format(self.optname, self.ofname)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])

class SubStar(Dao):
	def __init__(self, opts):
		super(SubStar, self).__init__('daophot', opts)
		
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('daophot', general_name=True)
		self.write_opts('daophot', pars=self.opts['daophot'])
		
		# build call
		lines = """options
{0}

attach {1}.fits

substar
{1}.psf
{1}.nst
y
{1}.lst
{1}.sub

exit
END_DAOPHOT""".format(self.optname, self.ofname)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])

class Allstar(Dao):
	def __init__(self, opts):
		super(Allstar, self).__init__('allstar', opts)
	
	def __call__(self, fname):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('allstar', pars=self.opts['allstar'], general_name=True)
		
		lines = """

{0}.fits
{0}.psf
{0}.ap
{0}.als
{0}.sub.fits
""".format(self.ofname)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "END_DAOPHOT"])


# general helper functions
def rm_list(fname, bad_list, length=None):
	"""Remove elements of bad_list from file fname
	Assumes elements of bad_list are strings"""
	
	# find string length
	if length==None:
		length = len(bad_list[0])
		
	out = open(fname+'tmp', 'w')
	f = open(fname, 'r')
	for line in f:
		if line[:7] not in bad_list:
			out.write(line)
	f.close()
	out.close()
	move(fname+'tmp', fname)

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

# testing
def test(fname):
	"""Testing new photometric pipeline"""
	
	# global options
	opts = {'daophot': {'r_psf': 20, 'r_fit': 15, 'fwhm': 3.5, 'psf_model': 5.00}, 'photo': {}, 'allstar': {}, 'allframe': {}, 'misc': {'stacknum': 1, 'counts_limit': 15000, 'number_limit': 400, 'sigma_psf': 4.5, 'sigma_all': 3.5}}
	
	# image
	#fname = "test.fits"
	name = os.path.splitext(fname)[0]
	dr = os.path.split(fname)[0]
	ofname = os.path.basename(name)
	
	# individual options
	set_indopts(fname, opts)
	
	# psf image (masked image)
	os.system("python psf_mask.py {0} {1} {2} {3}".format(fname, opts['misc']['counts_limit'], opts['misc']['number_limit'], opts['daophot']['high_good']+1))
	psfname = name + "_psf.fits"
	
	# if psf image not created, copy the original image
	if os.path.isfile(psfname)==False:
		copy(fname, psfname)
		
	# cleanup
	cleanup_files(fname, ("log", "coo", "ap", "psf", "inp", "als", "sub.fits"))
	cleanup_files(psfname, ("log", "coo", "ap", "nei", "psf", "grp", "nst", "lst", "sub.fits"))
	
	# create photometry objects
	finder = Find(opts)
	aper = Aper(opts)
	pick = PickStars(opts)
	psf = GetPSF(opts)
	group = GroupStars(opts)
	nstar = NStar(opts)
	sub = SubStar(opts)
	astar = Allstar(opts)

	# general options
	opts['daophot']['sigma_th']=opts['misc']['sigma_all']

	# initial find, to get fwhm
	finder(fname)
	aper(fname)
	
	# once initial find run, determine fwhm, update it in opts
	
	# options for psf finding
	opts['daophot']['sigma_th']=opts['misc']['sigma_psf']
	
	# aperture photometry
	finder(psfname)
	aper(psfname)
	
	# build psf
	print('building psf ...')
	pick(psfname)
	bad_list = psf(psfname)
	
	# remove bad stars from list
	while bad_list:
		rm_list(name+"_psf.lst", bad_list)
		bad_list = psf(psfname)
	
	# create star groups
	group(psfname)
	
	# subtract neighbors
	nstar(psfname)
	sub(psfname)
	
	# if subtracted image not in current directory, move there
	#if os.path.isfile(name+"_psfs.fits")==False:
		#print("moving s.fits")
	#	move("s.fits", name+"_psfs.fits")
	
	# use neighbor-subtracted image for measuring psf
	move(name+"_psf.fits", name+"_psf_old.fits")
	move(name+"_psf.sub", psfname)
	
	# final psf
	bad_list = psf(psfname)
	
	# remove bad stars from list
	while bad_list:
		rm_list(name+"_psf.lst", bad_list)
		bad_list = psf(psfname)
	
	# rename psf
	move(name+"_psf.psf", name+".psf")
	
	# run allstar
	print('running allstar ...')
	astar(fname)
	
	# remove options files
	trash = glob.glob(dr+'/*%s*.opt'%ofname)
	#trash.extend(glob.glob('*.opt'))
	trash.extend([name+"_psf.fits", name+"_psf_old.fits"])
	
	for t in trash:
		silent_remove(t)
	
def test_dirs():
	"""test code in a different directory"""
	
	# dir, caution, dir can't have too long name
	dr = "test/"
	f = glob.glob(dr+'*_?.fits')
	f = sorted(f)
	#print(f)
	
	for fname in f:
		print(fname)
		test(fname)
	
	
	
	
	
	



