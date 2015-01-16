from __future__ import print_function, division

import os
from shutil import copy, move
import subprocess
from subprocess import call, Popen, PIPE
import glob
import string
import pickle

import numpy as np
from astropy.io import fits
from scipy.optimize import leastsq

pi=3.14159265359

# define options keys and default values for 4 types of options files
OPTS_DEFAULT = {'daophot': {'read_noise': 6.70, 'gain': 4.20, 'low_good': 7.00, 'high_good': 32766.50, 'fwhm': 2.50, 'sigma_th': 4.00, 'low_sharp': 0.30, 'high_sharp':1.40, 'low_round': -1.00, 'high_round':1.00, 'watch':0.00, 'r_fit': 15.00, 'r_psf': 20.00, 'var_psf': 2.00, 'sky': 0.00, 'psf_model': 1.00, 'psf_clean': 5.00, 'use_sat': 0.00, 'err_percent': 0.75, 'err_profile': 5.00},
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
		self.opts['misc']['maglim'] = float(stdout[s1:s1+4])-2.

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
			#rvar = ["%7s"%(profile_errors[i - 1]) for i, x in enumerate(profile_errors) if (float(x) > 0.05) & (float(x) < 1.)]
			
			# only return bad_psf list if a required minimum of psf stars remains
			fine = 0.5*(len(profile_errors) - 2*len(sat) - 3*len(var))
			if fine>self.opts['misc']['minpsf']:
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

class Daomatch(Dao):
	def __init__(self, opts):
		super(Daomatch, self).__init__('daomatch', opts)
	
	def __call__(self, fname, fin):
		"""Call daophot and run find command"""
		self.init_names(fname)
		
		auxlines = ""
		for x in fin[1:]:
			auxlines+="{0}\ny\n".format(x)
		
		lines = """{0}
{1}
{2}

EOF""".format(fin[0], self.ofname, auxlines)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "EOF"])

class Daomaster(Dao):
	def __init__(self, opts):
		super(Daomaster, self).__init__('daomaster', opts)
	
	def __call__(self, fname, nfiles, index, idtrans):
		"""Call daophot and run find command"""
		self.init_names(fname)
		
		# matching radius countdown
		auxlines = "{0}\n".format(self.opts['misc']['r_match'])
		for x in range(nfiles-1):
			auxlines+="\n"
		
		for x in range(self.opts['misc']['r_match']-1, 2, -1):
			auxlines+="{0}\n".format(x)
		
		for x in range(self.opts['misc']['repeat_2']):
			auxlines+="2\n"
			
		for x in range(self.opts['misc']['repeat_1']):
			auxlines+="1\n"
			
		auxlines+="0"
		
		lines = """{0}.mch
{1}, {2}, {3}
{4}
{5}
{6}
y
y
{0}.mag{7:1d}
y
{0}.cor{7:1d}
y
{0}.raw{7:1d}
y
{0}.mch{7:1d}
n
n
{8}
EOF""".format(self.ofname, self.opts['misc']['min_frame'], self.opts['misc']['min_frac'], self.opts['misc']['enough_frame'], self.opts['misc']['max_sigma'], self.opts['misc']['degfree'], auxlines, index, idtrans)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "EOF"])

class Allframe(Dao):
	def __init__(self, opts):
		super(Allframe, self).__init__('allframe', opts)
	
	def __call__(self, fname, index):
		"""Call daophot and run find command"""
		self.init_names(fname)
		self.write_opts('allframe', pars=self.opts['allframe'], general_name=True)
		
		lines = """
{0}.mch{1:1d}
{0}.mag{1:1d}
EOF
""".format(self.ofname, index)
		
		# call daophot cmd
		self.run_cmd(lines, args=["<<", "EOF"])

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
	
	# daophot expects readnoise in units of ADU, while header provides in units of electrons
	opts['daophot']['read_noise'] /= opts['daophot']['gain']

def lines_startwith(fin, begin, fout, toggle=True, skip=0):
	"""print lines from file <fin> starting with string <begin>"""
	out = open(fout, 'w')
	f = open(fin, 'r')
	
	count=0
	for line in f:
		if count>=skip:
			if line.startswith(begin) == toggle:
				out.write(line.lstrip())
				#print(line.lstrip(), end="")
		count+=1
	f.close()
	out.close()

def hmerge_files(files, fout):
	"""Horizontally merge files
	Parameters:
	files - tuple of filenames to be merged
	fout - name of the output file"""
	
	# number of files
	nfile=len(files)
	f = [open(fin, 'r') for fin in files]
	out = open(fout, 'w')
	
	# number of lines in files
	nline = len(list(f[0]))
	f[0].seek(0, 0)
	
	# merge line by line
	for i in range(nline):
		line = ""
		line += f[0].readline().rstrip('\n')
		for i in range(1,nfile):
			line += "  "
			line += f[i].readline().rstrip('\n')
		line += '\n'
		out.write(line)
		
	# close
	for i in range(nfile):
		f[i].close()
	out.close()
	
def get_fwhm(name, rfit=20):
	ids, x, y, dm, sharp, rnd = load_starlist(name+".coo")
	
	# select bright stars
	bright = (dm<-1.5) & (sharp<1) & (sharp>0.5) & (rnd<0.3) & (rnd>-0.3)
	
	# relax roundness constraint if too few stars selected
	if np.sum(bright)<50:
		bright = (dm<-1.5) & (sharp<1) & (sharp>0.5) & (rnd<0.7) & (rnd>-0.7)
		
	xb = x[bright]
	yb = y[bright]
	Nb = np.size(xb)
	fwhm = np.zeros((Nb, 2))
	
	# load data
	hdulist = fits.open(name+".fits")
	data = hdulist[0].data
	
	for i in range(Nb):
		fwhm[i] = fitgaussian(data, xb, yb, rfit, i)
	
	hdulist.close()
	
	# median fwhm for bright stars (along x & y directions)
	fwhm_med = np.median(fwhm)
	
	## make sure fwhm is reasonable
	#if (fwhm_med<2) or (fwhm_med>20):
		#fwhm_med = 5.
	
	return(fwhm_med)

def load_starlist(fcoo):
	"""Return starlist for the image"""
	ids, x, y, dm, sharp, rnd = np.loadtxt(fcoo, skiprows=3, usecols=(0,1,2,3,4,5), unpack=True)
	x -= 1
	y -= 1
	return (ids, x, y, dm, sharp, rnd)

def fitgaussian(fulldata, xs, ys, rfit, n):
	"""Returns (height, x, y, width_x, width_y, const)
	the gaussian parameters of a 2D distribution found by a fit"""
	Ny, Nx = np.shape(fulldata)
	xc, yc = (xs[n], ys[n])
	
	x0 = xc - rfit
	x1 = xc + rfit
	y0 = yc - rfit
	y1 = yc + rfit
	
	if x0<0:
		x0=0
	if y0<0:
		y0=0
	if x1>=Nx:
		x1=Nx-1
	if y1>=Ny:
		y1=Ny-1
	
	#print(xc, yc, rfit, x0, x1, y0, y1)
	data=fulldata[y0:y1,x0:x1]
	
	#initial guesses
	params = [np.max(data), rfit, rfit, 0.1*rfit, 0.1*rfit, np.median(data)]
	
	errorfunction = lambda p: np.ravel(gaussian(*p[:5])(*np.indices(data.shape)) + params[-1] - data)
	p, success = leastsq(errorfunction, params)
	fwhm = 2*np.sqrt(2.*np.log(2.))*p[3:5]
	return np.abs(fwhm)

def gaussian(height, center_x, center_y, width_x, width_y):
	"""Returns a gaussian function with the given parameters"""
	width_x = float(width_x)
	width_y = float(width_y)
	return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def loaddaotxt(fname, usecols=None, unpack=False):
	"""Read aperture photometry file"""
	
	f = open(fname, 'r')
	
	# skip first line
	f.readline()
	
	# get number of lines per star
	line = f.readline()
	Nl = int(string2array(line)[0])
	dline = [None]*Nl
	
	# skip empty lines
	l1 = f.readline()
	while len(l1.replace(" ","").replace("\n",""))==0:
		l1 = f.readline()
	
	# load first data line
	d1 = string2array(l1)
	l2 = f.readline()
	d2 = string2array(l2)
	data = np.concatenate((d1,d2))
	data = np.array([data])
	
	# load subsequent data lines
	l = 0
	d = np.empty(0)
	
	for line in f:
		
		#only load non-empty lines
		if len(line.replace("\n",""))>0:
			ind = l%Nl
			dline[ind] = string2array(line)
			
			if ind==Nl-1:
				# combine parts of the line
				for i in range(Nl):
					d = np.concatenate((d,dline[i]))
				d = [d]
				
				# add line to the data matrix
				data = np.concatenate((data,d),axis=0)
				d = np.empty(0)
			l += 1
	f.close()
	
	# transpose data matrix
	if unpack==True:
		data = data.T
	
	# extract requested columns
	if usecols!=None:
		retdat = []
		for c in usecols:
			retdat.append(data[c])
	else:
		retdat = data
	
	return retdat

def string2array(line):
	"""Transform a list of strings into a numpy array"""
	
	# remove spaces and return
	lst = str.split(line.replace("\n","")," ")
	lst = filter(lambda a: a != "", lst)
	
	# cast on float
	lst = [float(x) for x in lst]
	
	return np.array(lst)

# testing
def test_phot(fname):
	"""Testing new photometric pipeline"""

	# image names
	#fname = "test.fits"
	name = os.path.splitext(fname)[0]
	dr = os.path.split(fname)[0]
	ofname = os.path.basename(name)
	
	# global options
	opts = {'daophot': {'r_psf': 25, 'r_fit': 20, 'fwhm': 3.5, 'psf_model': 7.00, 'var_psf':0., 'psf_clean':50., 'err_percent': 0.75, 'err_profile': 5.00}, 'photo': {}, 'allstar': {}, 'allframe': {}, 'misc': {'stacknum': 1, 'counts_limit': 15000, 'number_limit': 400, 'minpsf': 35, 'sigma_psf': 10., 'sigma_all': 3., 'r_see': 20}}
	
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
	
	#opts['misc']['maglim']=15.5
	#print(opts['misc']['maglim'])
	
	# measure seeing and update daophot fwhm
	opts['daophot']['fwhm'] = get_fwhm(name, rfit=opts['misc']['r_see'])
	
	# save options to file
	pickle.dump(opts, open(name+".opts", "wb" ) )
	
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
	#copy(name+".psf", name+".psf_orig")
	#move(name+".psf_edit", name+".psf")
	#copy("/home/ana/projects/pal5/data/images/09/01/0/"+ofname+".psf", name+".psf")
	
	# run allstar
	print('running allstar ...')
	astar(fname)
	
	# remove options files
	trash = glob.glob(dr+'/*%s*.opt'%ofname)
	#trash.extend(glob.glob('*.opt'))
	trash.extend([name+"_psf.fits", name+"_psf_old.fits"])
	
	for t in trash:
		silent_remove(t)

def test_match(fname):
	"""Test source matching with allframe"""
	# global options
	opts = {'daophot': {'r_psf': 20, 'r_fit': 15, 'fwhm': 3.5, 'psf_model': 5.00}, 'photo': {}, 'allstar': {}, 'allframe': {}, 'misc': {'stacknum': 1, 'counts_limit': 15000, 'number_limit': 400, 'minpsf': 35, 'sigma_psf': 4.5, 'sigma_all': 3.5, 'min_frame': 1, 'min_frac': 0.1, 'enough_frame': 2, 'max_sigma': 10, 'degfree': 6, 'r_match': 15, 'repeat_2': 3, 'repeat_1': 30}}
	
	# cleanup
	#cleanup_files(fname, ("log", "mch", "mag1", "raw1", "mch1", "cor1"))
	#cleanup_files(fname, ("mtr", "mag2", "raw2", "mch2", "cor2"))
	
	# initialize names
	name = os.path.splitext(fname)[0]
	dr = os.path.split(fname)[0]
	ofname = os.path.basename(name)
	
	# compile catalog list
	catalogs = glob.glob(dr+"/*.als")
	catalogs = sorted(catalogs)
	oldindex=[ i for i, x in enumerate(catalogs) if "r_1" in x][0]
	catalogs.insert(0, catalogs.pop(oldindex))
	catalogs = [os.path.basename(x) for x in catalogs]
	nfiles = len(catalogs)
	
	## run daomatch 
	#match = Daomatch(opts)
	#match(fname, catalogs)
	
	## run daomaster
	#master = Daomaster(opts)
	#master(fname, nfiles, index=1, idtrans='n')
	
	## run allframe
	#aframe = Allframe(opts)
	#aframe(fname, index=1)
	
	## first move to targeted directory
	#origwd = os.getcwd()
	#os.chdir(dr)
	
	## get & sort catalogs
	#catalogs = glob.glob("*.alf")
	#catalogs = sorted(catalogs)
	#oldindex=[ i for i, x in enumerate(catalogs) if "r_1" in x][0]
	#catalogs.insert(0, catalogs.pop(oldindex))
	#nfiles = len(catalogs)
	
	## change back directory
	#os.chdir(origwd)
	
	## run daomatch 
	#match = Daomatch(opts)
	#match(fname, catalogs)
	
	## run daomaster
	#master = Daomaster(opts)
	#master(fname, nfiles, index=2, idtrans='y')
	
	# final catalog
	lines_startwith(name+".raw2", "              ", fout=name+"_cs.temp")
	lines_startwith(name+".raw2", "              ", fout=name+"_mag.temp", toggle=False, skip=2)
	
	hmerge_files((name+"_mag.temp", name+"_cs.temp"), fout=name+".cat")

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

import glob
def scratch():
	f = glob.glob("test/*opt")
	f = sorted(f)
	print(f)
	
	base = [os.path.splitext(x)[0] for x in f]
	print(base)
	
	
	
	
	



