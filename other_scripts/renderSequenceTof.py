''' script for generating Luxrender lxs files for ToF Simulation'''
''' Stephan Meister <stephan.meister@iwr.uni-heidelberg.de> 
	Heidelberg Collaboratory for Image Processing (HCI)
	Interdisciplinary Institute for Scientific Computing (IWR)
	University of Heidelberg, 2013'''


import re
import sys	
import math
import os
import subprocess
import glob
# import h5py
# import time

def s2Int(s):
	if s[-1] in ["M","m"]:
		return int(s[:-1]) * 1000000
	if s[-1] in ["K","k"]:
		return int(s[:-1]) * 1000
	return int(s)

def int2S(i):
	if i // 1000000 > 0:
		return "%uM" % (i // 1000000)
	if i // 1000 > 0:
		return "%uk" % (i // 1000)
	return "%u" % i

def getVal(default):
	r = input().strip()
	if r == '':
		return default
	else:
		return r
	
def replaceOrInsert(value, section, target):
	if re.search(value,target):
		return target
	else:
		target = re.sub(section, '''%s\n\t%s''' % (section, value),target)
	return target

luxconsole = r"C:\Users\Administrador\Documents\MultipathTofSimulator\win32\luxconsole.exe"
exr2depth = r"C:\Users\Administrador\Documents\MultipathTofSimulator\exr2depth\bin\exr2depth.exe"

if __name__ == "__main__":
	shutdown = False
	try:

		print("Datapath []: ",end="")
		datapath = getVal("").replace('''\\''','''/''')

		lxs_file_list = sorted(glob.glob(os.path.join(datapath, "*.lxs")))
		positions = []
		
		# --------------------------------- lxs file template ------------------------
		
		print("Eye depth [6]: ", end="")
		eyeDepth = s2Int(getVal("6"))
		print("Light depth [6]: ", end="")
		lightDepth = s2Int(getVal("6"))
		print("Samples [1k]: ",end="")
		samples = s2Int(getVal("1k"))
		print("ToF Freq [20M]: ",end="")
		toffreq = s2Int(getVal("20M"))
		print("Harmonics [0]: ", end="")
		harmonics = s2Int(getVal("0"))
		print("Harmonics shift[0.0]: ", end="")
		harmonicsShift = (getVal("0.0"))
		print("Harmonics Intensity[0.0]: ", end="")
		harmonicsInt = (getVal("0.0"))

		print("Render output directory [sequence]: ",end="")
		render_outdir = getVal("sequence").replace('''\\''','''/''')
		print("lxs file prefix [image]: ",end="")
		file_prefix = getVal("image")
		print("shutdown? y/n [n]: ",end="")
		shutdown = getVal("n") == "y"

		if shutdown:
			print("WILL SHUTDOWN AT END")
		
		if not render_outdir == "" and not os.path.exists(render_outdir):
			os.makedirs(render_outdir)
		
		for file_i,file in enumerate(lxs_file_list):
			
			#Create frame directory
			if not os.path.exists(os.path.join(render_outdir, f"frame{file_i}")):
				os.makedirs(os.path.join(render_outdir, f"frame{file_i}"))

			f = open(file)
			template = f.read()
			f.close()

			#Add ground truth
			positions.append(re.findall(r"LookAt (.+)", template)[0] + "\n")
			
			p = re.compile(r'''eyedepth" \[[0-9]+\]''')
			template = re.sub(p,'''eyedepth" [%u]''' % eyeDepth,template,count = 1)
			p = re.compile(r'''lightdepth" \[[0-9]+\]''')
			template = re.sub(p,'''lightdepth" [%u]''' % lightDepth,template,count = 1)
			p = re.compile(r'''haltspp" \[[0-9]+\]''')
			template = re.sub(p,'''haltspp" [%u]''' % samples,template,count = 1)

			p = re.compile('''"integer displayinterval" \[[0-9]+\]''')
			template = re.sub(p,'''"integer displayinterval" [60]''',template,count = 1)
			p = re.compile('''"integer writeinterval" \[[0-9]+\]''')
			template = re.sub(p,'''"integer writeinterval" [60]''',template,count = 1)
			p = re.compile('''"integer flmwriteinterval" \[[0-9]+\]''')
			template = re.sub(p,'''"integer flmwriteinterval" [120]''',template,count = 1)

			template = replaceOrInsert('''"float toffreq" [%s]''' % toffreq,'''SurfaceIntegrator "bidirectional"''',template)
			template = replaceOrInsert('''"float modint" [0.75]''','''SurfaceIntegrator "bidirectional"''',template)
			template = replaceOrInsert('''"float modoffset" [0.25]''','''SurfaceIntegrator "bidirectional"''',template)
			template = replaceOrInsert('''"float phaseshift" [0]''','''SurfaceIntegrator "bidirectional"''',template)
			template = replaceOrInsert('''"integer harmonics" [%s]'''% harmonics,'''SurfaceIntegrator "bidirectional"''',template)
			template = replaceOrInsert('''"float harmonicsint" [%s]'''% harmonicsInt,'''SurfaceIntegrator "bidirectional"''',template)
			template = replaceOrInsert('''"float harmonicsshift" [%s]'''% harmonicsShift,'''SurfaceIntegrator "bidirectional"''',template)
			
			
			phaseshifts = [0, 0.5 * math.pi, 1.0 * math.pi, 1.5 * math.pi]

			lxsFiles = []
			exr_files = []
			
			for i in [0,1,2,3]:
				p = re.compile(r'''filename" \[".*?"\]''')
				exr_filepath = "%s_phase%u" % (file_prefix, i)
				exr_files.append(exr_filepath+".exr")
				template = re.sub(p,'''filename" ["%s"]''' %(exr_filepath), template,count = 1)
				
				p = re.compile(r'''phaseshift" \[.*?]''')
				template = re.sub(p,'''phaseshift" [%f]''' % phaseshifts[i],template,count = 1)
				
				filename = "%s/%s_%u.lxs" % (datapath,file_prefix,i)
				out = open(filename,"w")
				out.write(template)
				out.close()
				lxsFiles.append(filename)
			if len(sys.argv) > 2 and sys.argv[2] == "--norun":
				exit()
			
			#----------------------------- Run Renderer -----------------------
			
			for i in [0,1,2,3]:
				print("\n\nRunning Luxrender on file %i\n" % i)
				p = subprocess.Popen([luxconsole, lxsFiles[i]])
				p.wait()
			
			# p = subprocess.Popen([exr2depth, exr_files[0], exr_files[1], exr_files[2], exr_files[3], f"{render_outdir}/{file_prefix}{file_i}", f"{toffreq//1000000}Mhz", "bmp"])
			# p.wait()
			
			#Delete generated files
			
			for del_file in lxsFiles:
				os.remove(del_file)
			# for del_file in exr_files:
			# 	os.remove(del_file)

			# Move exr files
			for exr_file in exr_files:
				os.rename(f"{datapath}/{exr_file}", f"{render_outdir}/frame{file_i}/{exr_file}")

		print("\nfinished")

		with open(os.path.join(render_outdir,"ground_truth.txt"), "w") as gt_file:
			gt_file.writelines(positions)
	
	except Exception as e:
		print("ERROR:",str(e))

	finally:
		if shutdown:
			p = subprocess.Popen(["shutdown", "/s"])