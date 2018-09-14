
# coding: utf-8

# MUST SOURCE ACTIVATE spectralDNS

# In[1]:


import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import pyqg


# In[2]:


# create the model object
m = pyqg.BTModel(L=2.*np.pi, nx=256, #128, #64, #nx=256,
                 beta=0., H=1., rek=0., rd=None,
                 #tmax=40, dt=0.001, taveint=1,
                 tmax=160, dt=0.001, taveint=1,
                 ntd=4)
# in this example we used ntd=4, four threads
# if your machine has more (or fewer) cores available, you could try changing it


# In[3]:


# generate McWilliams 84 IC condition

fk = m.wv != 0
ckappa = np.zeros_like(m.wv2)
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

nhx,nhy = m.wv2.shape

Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

Pi = m.ifft( Pi_hat[np.newaxis,:,:] )
Pi = Pi - Pi.mean()
Pi_hat = m.fft( Pi )
KEaux = m.spec_var( m.wv*Pi_hat )

pih = ( Pi_hat/np.sqrt(KEaux) )
qih = -m.wv2*pih
qi = m.ifft(qih)


# In[4]:


# initialize the model with that initial condition
m.set_q(qi)


# In[5]:

# In[7]:

# In[8]:



# # Build a datatset

# In[47]:


N_samples=2000 #10#100
nx=256 #128 #64 #256 # linear resolution

tmax=40
tsnapint=10

vort_stored=np.zeros((N_samples, tmax//tsnapint, nx, nx))


# create the model object
m = pyqg.BTModel(L=2.*np.pi, nx=nx,
                 beta=0., H=1., rek=0., rd=None,
                 tmax=tmax, dt=0.001, taveint=1,
                 ntd=4)



print(vort_stored.shape)


# In[48]:


# generate McWilliams 84 IC condition

def generate_ic():
    fk = m.wv != 0
    ckappa = np.zeros_like(m.wv2)
    ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1

    nhx,nhy = m.wv2.shape

    Pi_hat = np.random.randn(nhx,nhy)*ckappa +1j*np.random.randn(nhx,nhy)*ckappa

    Pi = m.ifft( Pi_hat[np.newaxis,:,:] )
    Pi = Pi - Pi.mean()
    Pi_hat = m.fft( Pi )
    KEaux = m.spec_var( m.wv*Pi_hat )

    pih = ( Pi_hat/np.sqrt(KEaux) )
    qih = -m.wv2*pih
    qi = m.ifft(qih)

    return(qi)


# In[49]:


for i_samp in range(N_samples):
    print('Generating sample/total', i_samp+1, N_samples)

    # create the model object (each round to reset timer for snapshots)
    m = pyqg.BTModel(L=2.*np.pi, nx=nx,
                 beta=0., H=1., rek=0., rd=None,
                 tmax=tmax, dt=0.001, taveint=1,
                 ntd=4)
    
    
    qi=generate_ic()
    m.set_q(qi)
    #plot_q(m) # plot IC of each sample

    # Use a try/except clause to ignore rare CFL crashes
    try:
        for j_snap, _ in enumerate(m.run_with_snapshots(tsnapstart=0, tsnapint=tsnapint)):
            vort_stored[i_samp,j_snap,:,:]=m.q.squeeze()
    except:
        vort_stored[i_samp,:,:,:]= vort_stored[i_samp-1,:,:,:] # repeat previous run results when model crashed because of CFL


print(vort_stored.shape)


# In[52]:

np.save('vort_stored_nx256_x.npy', vort_stored)

print('Done, saved in npy file')

