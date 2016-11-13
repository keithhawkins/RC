#matplotlib notebook
import pystan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from astropy.table import Table


datapath='/Users/khawkins/Desktop/RAVE_cannon/GaiaDR1/' # defines the path the data is stored


def load_data(typ='fake', SNRpar = 10, mag_err_cut = 0.5,plots=True):
	#---- define the photometric band
	photband = 'K'
	photband_err = 'K_ERR'

	if typ == 'fake':
		#----use this for the fake sample for testing only

		# Generate some fake data with the right equations
		numobj = 250  # number of objects
		M = -1.62  # absolute mag
		sigma_M = 0.1  # stddev of M in the population
		d = np.random.uniform(low=10, high=1e2, size=numobj)  # draw distance, in units of 10 pc
		G_true = M + 5*np.log10(d) + sigma_M * np.random.randn(numobj)  # true G magnitude, include M scatter
		sigma_G = np.random.uniform(low=0.01, high=0.2, size=numobj)  # rror on observed G magnitudes
		G = G_true + sigma_G * np.random.randn(numobj)  # draw noisy magnitudes
		omega_true = (1. / (d*10))*1000.0 # compute true parallax omega from d, times 10 pc
		sigma_omega = 0.01 * omega_true  # error on omega
		omega = omega_true + sigma_omega * np.random.randn(numobj)  # noisy parallax omega
	


	if typ == 'APOKASC':
		#----use this for the APOKASC sample 
		L = Table.read(datapath+'APOKASK_TGAS.fits') 
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		paruncert=  sigma_omega/omega
		ind = np.where((omega >= 0)&(L['stage'] == 'RC   ')&(L[photband_err] < mag_err_cut  ) & (paruncert < 1./SNRpar) )[0]
		#----redefine the table with only the 'cleaned' sample
		L = L[ind] 
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		G = L[photband] ; sigma_G = L[photband_err] 

	elif typ == 'Bovy':
		#----use this for the Bovy/APOGEE DR13 sample 
		L = Table.read(datapath+'BovyRC_TGAS.fits') 
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		paruncert=  sigma_omega/omega
		ind = np.where((omega >= 0)&(L[photband_err] < mag_err_cut  ) & (paruncert < 1./SNRpar) )[0]
		#----redefine the table with only the 'cleaned' sample
		L = L[ind] 
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		G = L[photband] ; sigma_G = L[photband_err] 


	elif typ == 'Laney':
		L = Table.read('/Users/khawkins/Desktop/RAVE_cannon/GaiaDR1/Laney2012_TGAS.fits')
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		paruncert=  sigma_omega/omega
		ind = np.where((omega >= 0)&(L[photband_err] < mag_err_cut  ) & (paruncert < 1./SNRpar) )[0]
		#----redefine the table with only the 'cleaned' sample
		L = L[ind] 
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		G = L[photband] ; sigma_G = L[photband_err] 

	elif typ=='APO1m':
		#-----use this for the Fullet APO 1-m RC sample
		L = Table.read('/Users/khawkins/Desktop/RAVE_cannon/GaiaDR1/APOGEE_TGAS_RC_new.fits')
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		paruncert=  sigma_omega/omega
		ind = np.where((omega >= 0)&(L[photband_err] <= mag_err_cut  ) & (paruncert <= 1./SNRpar) )[0]
		#----redefine the table with only the 'cleaned' sample
		L = L[ind] 
		omega = L['parallax'] ; sigma_omega = L['parallax_error'] 
		G = L[photband] ; sigma_G = L[photband_err] 

	else:
		raise ValueError('Please replace typ to : APOGEE1m, Bovy, Laney, or APOKASC')



	if plots:
		#make a few plots of the observed quantities
		fig, axs = plt.subplots(1, 3, figsize=(11, 11))
		axs = axs.ravel()
		sns.kdeplot(G, ax=axs[0])
		axs[0].set_xlabel('K')
		sns.kdeplot(omega, ax=axs[1])
		axs[1].set_xlabel(r'$\varpi$ (mas)')
		sns.kdeplot(1./(omega/1.E2), ax=axs[2])
		axs[2].set_xlabel('d/(10pc)')
		#axs[2].hist(1./(omega/1.E2))
		fig.tight_layout()

	return G,sigma_G,omega,sigma_omega, L 


def run_stan(G, sigma_G, omega,sigma_omega,niters=10000, numchains=2 ):
	numobj = len(omega)
	stan_code = """
        
	functions
	{
	   real foo_lpdf(real y, real L)
	   {
	    
	       #return log(y*y/(0.333333*pow(400,3))); #defines a uniform space density prior within 10pc - 4kpc
	       return log( (1/(2*exp(3*L))) * (y*y)*exp(-y/exp(L))); #defines an exponentially decreasing distance prior with scale-length, L
	   }
	}
	
	data {
	    int<lower = 0> N; 
	    real<lower = 0> omega[N];
	    real<lower = 0> sigma_omega[N];
	    real<lower = -10> G[N];
	    real<lower = 0> sigma_G[N];
	    
	}
	parameters {
	    real<lower=-8, upper=5> M;
	    real<lower=0, upper=1> sigma_M;
	    real<lower=1.0, upper = 400> d[N];
	    real<lower = log(1), upper =log(400)> L;
	}
	transformed parameters {
	    real<lower=0> omega_true[N];
	    real<lower=-10> G_true[N];
	    real<lower=0> sigma_G_tot[N];
	    for (n in 1:N){
	        G_true[n] <- M + 5 * log10(d[n]);
	        sigma_G_tot[n] <- sqrt(pow(sigma_G[n], 2) + pow(sigma_M, 2));
	        omega_true[n] <- 1000*(1. / (d[n]*10));
	    }
	}
	model {
	    # need to put priors on M, sigma_M, and d
	    # e.g. M ~ normal ...
	    # e.g. sigma_M ~ gamma ...
	    # e.g. p(d) propto d^2 ? ...
	    G ~ normal(G_true, sigma_G_tot);
	    omega ~ normal(omega_true, sigma_omega);
	    sigma_G ~uniform(0,5);
	    L ~ uniform(log(1),log(500));
	    #d[N] ~  exponential(1/L);
	    for (i in 1:N)
	    {
	    d[i]~foo(L);
	    }
	    
	    
	}
	"""
	 
	# make a dictionary containing all data to be passed to STAN
	stan_dat = {
	    'N': numobj, 
	    'omega': omega,
	    'sigma_omega': sigma_omega,
	    'G': G,
	    'sigma_G': sigma_G
	}
	 
	# Fit the model
	fit = pystan.stan(model_code=stan_code, data=stan_dat, iter=niters, chains=numchains)

	return fit,stan_dat


def plot_chains(fit,stan_dat):
	omega = stan_dat['omega'] ; sigma_omega = stan_dat['sigma_omega']; G = stan_dat['G']; signa_G = stan_dat['sigma_G']
	# ---- plot the chains and best estimates for the M, sigma M 
	M = np.percentile(fit.extract('M')['M'],50)
	Mp = np.percentile(fit.extract('M')['M'],84)- M
	Mm =  M-np.percentile(fit.extract('M')['M'],16)
	
	sigma_M = np.percentile(fit.extract('sigma_M')['sigma_M'],50)
	sigma_Mp = np.percentile(fit.extract('sigma_M')['sigma_M'],84) - sigma_M
	sigma_Mm= sigma_M-np.percentile(fit.extract('sigma_M')['sigma_M'],16)
	
	print 'M, sigma_M = %.2f (+%.2f,-%.2f), %.2f (+%.2f,-%.2f)'%(M,Mp,Mm, sigma_M,sigma_Mp,sigma_Mm)
	
	fig = fit.plot([r'M', r'sigma_M'])
	fig.axes[0].axvline(M, c='k', ls='--')
	fig.axes[1].axhline(M, c='k', ls='--')
	fig.axes[2].axvline(sigma_M, c='k', ls='--')
	fig.axes[3].axhline(sigma_M, c='k', ls='--')
	fig.figsize = (16, 16)
	fig.tight_layout()
	#---------------------


	#-----------fit the scale-length of the exponetially decreasing space density prior, comment out if using other priors
	plt.figure()
	sns.kdeplot(10.*np.exp(fit.extract('L')['L']))
	plt.xlabel('Scale-length (pc)')
	#-----------------------------


	#------ fit the first 16 stars inferred distance

	samples_d = fit.extract('d')['d']
	print np.shape(samples_d)
	# plot the posterior distributions and chains on the distances
	fig, axs = plt.subplots(4, 4, figsize=(12, 8))
	axs = axs.ravel()
	for i in range(axs.size):
	    sns.kdeplot(samples_d[:, i], ax=axs[i])
	    #axs[i].axvline(d[i], c='k', ls='--', lw=2, label='True distance')
	    axs[i].axvline(np.mean(samples_d[:, i]), c='b', ls='--', lw=2, label='HB estimate')
	    axs[i].axvline(1./(omega[i]/100.), c='r', ls='--', lw=2, label='Naive estimate')
	    snr_par = omega[i]/sigma_omega[i]
	    pnew = omega * (0.5 + 0.5*np.sqrt(1 - 16./snr_par**2))
	    axs[i].axvline(1./(pnew[i]/100.), c='c', ls='-.', lw=2, label='L-K correction')
	    axs[i].set_xlabel('d'+str(i))
	axs[0].legend(loc='upper left', frameon=False)
	fig.tight_layout()
	
	#----------------------------------------

	#----------plot the differences between the inferred, 1/parallax, and L-K correciton distances
	
	d_infer = []
	d_naive = []
	d_LK = []
	for i in np.arange(np.shape(samples_d)[1]):
	    d_infer.append(np.mean(samples_d[:, i]))
	    d_naive.append(1./(omega[i]/100.))
	    snr_par = omega[i]/sigma_omega[i]
	    pnew = omega[i] * (0.5 + 0.5*np.sqrt(1 - 16./snr_par**2))
	    d_LK.append(1./(pnew/100.))
	
	d_infer = np.array(d_infer); d_naive = np.array(d_naive) ; d_LK = np.array(d_LK)
	plt.figure()
	plt.plot(d_infer, d_naive-d_infer,'ks',label=r'1/$\varpi$')
	ind = np.isnan(d_LK)
	plt.plot(d_infer, d_LK-d_infer,'ro',label=r'L-K Correction')
	plt.axhline(y=0,color='k',ls='--',lw=2)
	plt.xlabel('d (/10pc)')
	plt.ylabel(r'$\Delta$d (/10pc)')
	plt.legend()
	

	plt.figure()
	plt.hist(d_naive-d_infer,histtype='step',lw=2,color='k',bins=20)
	plt.hist(d_LK[~ind]-d_infer[~ind],histtype='step',lw=2,color='r',bins=20)
	plt.xlabel(r'$\Delta$d (/10pc)')
	plt.text(1,120, '%.2f , %.2f (Naive) \n %.2f , %.2f (LK)'%(np.nanmean(d_naive-d_infer),\
	                                                         np.nanstd(d_naive-d_infer),
	                                                         np.nanmean(d_LK-d_infer),
	                                                         np.nanstd(d_LK-d_infer)))
	    
	print np.shape(fit.extract('M')['M'])		


	#------plot the inferred and niieve abs. mag distrubtion of the sample
	samples_d = fit.extract('d')['d']
	absMag = []
	absMag_inf = []
	for i in np.arange(np.shape(samples_d)[1]):
	    absMag.append(G[i]-5*np.log10(np.median(samples_d[:, i])))
	    dtest =1./(omega[i]/1.E2)
	    absMag_inf.append(G[i]-5*np.log10(dtest))
	
	plt.figure()    
	plt.hist(absMag,bins=20, histtype='step', label='Naive M',lw=3)
	plt.hist(absMag_inf,bins=20, histtype='step',label='inferred M',lw=3)
	plt.xlabel(r'M$_{K}$')
	plt.ylabel('N')
	plt.legend()
	# plot the posterior distributions and chains on the distances
	fig, axs = plt.subplots(4, 4, figsize=(12, 8))
	axs = axs.ravel()
	for i in range(axs.size):
	    sns.kdeplot(G[i]-5*np.log10(samples_d[:, i]), ax=axs[i])
	    #axs[i].axvline(d[i], c='k', ls='--', lw=2, label='True distance')
	    axs[i].axvline(G[i]-5*np.log10(np.mean(samples_d[:, i])), c='b', ls='--', lw=2, label='HB estimate')
	    axs[i].set_xlabel('G'+str(i))
	axs[0].legend(loc='upper left', frameon=False)
	fig.tight_layout()	

	#-----------------------


def paruncertainty_test(SNRpar=[1/0.1,1/0.2, 1/0.3, 1/0.5, 1/0.8,1. ], niters=20000,numchains=3):
	from scipy import stats
	SNRpar=[1/0.04, 1/0.06, 1/0.08, 1/0.1, 1/0.2 ]
	fits = []
	standats =[]
	fig, axs = plt.subplots(2, 2, figsize=(12, 8))
	fig.suptitle('Exponetially Decreasing Density Prior')
	axs = axs.ravel()
	colors = ['b','k','g','c','m','r','orange']
	for i in np.arange(len(SNRpar)):
		#load data and compute the model for each parallax uncertainty cut
		G,sigmaG,omega,sigmaomega = load_data(typ='APOGEE1m', SNRpar =SNRpar[i] , mag_err_cut = 0.2,plots=False)
		if len(G) < 10:
			print '---Warning: Not enough stars so skipping iteration where SNRpar = %2.f---'%SNRpar[i]
			continue

		fit,standat = run_stan(G,sigmaG,omega,sigmaomega,niters=niters,numchains=numchains)
		fits.append(fit) ; standats.append(standat)

		#compute the point estimates and uncertainties
		M = np.percentile(fit.extract('M')['M'],50)
		Mp = np.percentile(fit.extract('M')['M'],84)- M
		Mm =  M-np.percentile(fit.extract('M')['M'],16)
	
		sigma_M = np.percentile(fit.extract('sigma_M')['sigma_M'],50)
		sigma_Mp = np.percentile(fit.extract('sigma_M')['sigma_M'],84) - sigma_M
		sigma_Mm= sigma_M-np.percentile(fit.extract('sigma_M')['sigma_M'],16)

		#plotting everything

		sns.kdeplot(fit.extract('M')['M'],ax=axs[0], color=colors[i],label=r'$\sigma\varpi/\varpi$=%.2f, N=%i'%(1./SNRpar[i], len(G)))
		axs[0].axvline(x=M,color=colors[i],ls='--',lw=2)
		axs[0].set_xlabel(r'M')


		sns.kdeplot(fit.extract('sigma_M')['sigma_M'],ax=axs[1],color=colors[i])
		axs[1].axvline(x=sigma_M,color=colors[i],ls='--',lw=2)
		axs[1].set_xlabel(r'$\sigma$M')


		axs[2].plot(fit.extract('M')['M'],alpha=0.3,color=colors[i])
		axs[2].set_ylabel(r'M')
		axs[2].set_xlabel(r'Chain value')
		axs[3].plot(fit.extract('sigma_M')['sigma_M'],alpha=0.3,color=colors[i])
		axs[3].set_ylabel(r'$\sigma$M')
		axs[3].set_xlabel(r'Chain value')
		plt.legend()
	plt.tight_layout()
		


def make_HRD_colormag(tmass,SNRpar = 0.3):
	#tmass = Table.read('/Users/khawkins/Desktop/RAVE_cannon/GaiaDR1/2MASS_TGAS_30per.fits')

	g = tmass['G'] ; k =tmass['K']
	gmk = g-k

	MG = k + 5 * np.log10(tmass['parallax']) -10
	plt.figure(figsize=(12, 12))
	plt.hexbin(gmk,MG,cmap='gray',bins='log',mincnt=1e-4)
	
	plt.xlabel(r'$G-K_s$')
	plt.ylabel(r'M$_{K}$')


	sets = ['APO1m', 'Bovy', 'APOKASC','Laney'] ; symb = ['bs','r^','md','ko']
	for i in np.arange(len(sets)):
		dummy = load_data(typ=sets[i],SNRpar= 1./SNRpar,plots=False)
		L = dummy[-1] ; gs = L['phot_g_mean_mag'] ; ks = L['K']
		MGs = ks + 5 * np.log10(L['parallax']) -10
		plt.plot(gs-ks,MGs,symb[i],label='%s'%sets[i], alpha=0.8,mec='None')
		plt.xlim([-1.2,7])



	#lt.gca().invert_yaxis() 
	plt.axhline(y=-1.62,lw=2,ls='--',color='orange')
	plt.xlim([-0.23,4.51])
	plt.ylim([5.84,-6.])
	plt.title(r'$\sigma\varpi/\varpi <$%.2f'%SNRpar)
	plt.legend(loc='lower right',ncol=1, fontsize='large')
	plt.savefig('/Users/khawkins/Documents/HRD_colormag_samples_%iper.pdf'%(100*SNRpar),dpi=300)


















