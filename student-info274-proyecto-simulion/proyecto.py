import matplotlib.pyplot as plt
import jax.numpy as np
import jax.random as random
import numpyro
import numpy as n_p

import numpyro.distributions as dist

def sigmoid(z):
    return 1/(1+np.exp(-z))

def bayesian_neural_network(x, y=None, M: int = 1, sigma_prior=5.0):
    """
    Implemente el modelo de red neuronal bayesiana utilizando las primitivas de numpyro
    Nota: Puede agregar argumentos a la función si lo necesita
    """
    #Crea distribucion w,b donde son parámetros dependientes
    
    print("M: ",M," S_p:",sigma_prior)
    db_z = dist.Normal(loc=np.zeros(M), scale=5*np.ones(M)).to_event(1)
    dw_z = dist.Normal(loc=np.zeros(shape=(2,M)), scale=5*np.ones(shape=(2,M))).to_event(2)
    db_y = dist.Normal(loc=0, scale=5)
    dw_y = dist.Normal(loc=np.zeros((M,1)), scale=5*np.ones((M,1))).to_event(2)
        
    bz = numpyro.sample("bz", db_z)
    wz = numpyro.sample("wz", dw_z)
    by = numpyro.sample("by", db_y)
    wy = numpyro.sample("wy", dw_y)
   
    with numpyro.plate('datos', size=len(x)):
        
        #Enganchando capa oculta
        rst_z = np.dot(x,wz) + bz
        Z = sigmoid(rst_z)
                
        #Enganchando capa salida
        #rst_y = np.dot(Z,dw_y) + db_y
        u = numpyro.deterministic('u',np.sum(np.dot(Z, wy) + by, axis=1))
       
        #No se sabe si esta instruccion es funcional
        #logit_u = np.log(u_y/(1-u_y))
        Y_obs = numpyro.sample('y', dist.BernoulliLogits(u), obs=y)
        return Y_obs
            

def run_mcmc_nuts(partial_model, x, y, rng_key_):
    """
    Implemente una función que calcula y retorna la traza de los parámetros del modelo. 
    Utilice el algoritmo de muestreo No U-turn (NUTS)
    Nota: Puede agregar argumentos a la función si lo necesita
    """
    rng_key = random.PRNGKey(1234)
    rng_key, rng_key_ = random.split(rng_key)

    
    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.NUTS(partial_model), 
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=2)

    sampler.run(rng_key_, x, y)

    sampler.print_summary(prob=0.9)
    
    return sampler.get_samples()
    
    
    
    #rho = {}
    #rho['s'] = autocorrelation(posterior_samples['s'])
    #rho['w'] = autocorrelation(posterior_samples['theta'][:, 1])
    #rho['b'] = autocorrelation(posterior_samples['theta'][:, 0])
    
    
def run_mcmc_metropolis(partial_model, x, y, rng_key_):
    """
    Implemente una función que calcula y retorna la traza de los parámetros del modelo. 
    Utilice el algoritmo de muestreo Metropolis-Hastings de Barker (BarkerMH)
    Nota: Puede agregar argumentos a la función si lo necesita
    """
    rng_key = random.PRNGKey(1234)
    rng_key, rng_key_ = random.split(rng_key)

    sampler = numpyro.infer.MCMC(sampler=numpyro.infer.BarkerMH(partial_model), 
                             num_samples=1000, num_warmup=100, thinning=1,
                             num_chains=2)

    sampler.run(rng_key_, x, y)

    sampler.print_summary(prob=0.9)
    
    return sampler.get_samples()


def get_predictive_samples(partial_model,
                           x_test,
                           rng_key_,
                           mcmc_trace=None,
                           num_samples=None):
    """
    Retorna las muestras de la distribución predictiva
    """
    if mcmc_trace is not None:
        predictive = numpyro.infer.Predictive(partial_model,
                                              return_sites=(['u']),
                                              posterior_samples=mcmc_trace)
    elif num_samples is not None:
        print("Aqui 2")
        predictive = numpyro.infer.Predictive(partial_model,
                                              return_sites=(['y']),
                                              num_samples=num_samples)
    else:
        raise ValueError("Debe entregarse mcmc_trace o num_samples")
        
    return predictive(rng_key_, x_test)


def binary_mode(y):
    p = np.mean(y, axis=0)
    return p > 0.5


def entropy(y):
    p = np.mean(y, axis=0)
    return -p * np.log(p + 1e-10)


def plot_predictive_posterior(ax, x1, x2, y, cmap, vmin=0, vmax=1, title=None):
    cmap = ax.pcolormesh(x1,
                         x2,
                         y.reshape(len(x1), len(x2)),
                         cmap=cmap,
                         shading='gouraud',
                         vmin=vmin,
                         vmax=vmax)
    plt.colorbar(cmap, ax=ax)
    if title is not None:
        ax.set_title(title)


def plot_data(ax, x, y):
    ax.scatter(x[y == 0, 0], x[y == 0, 1], c='k', marker='o')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c='k', marker='x')
    ax.set_xlabel('x[:, 0]')
    ax.set_ylabel('x[:, 1]')


def autocorrelation(trace):
    """
    Retorna la autocorrelación de una traza
    """
    trace_norm = (trace - np.mean(trace)) / np.std(trace)
    rho = np.correlate(trace_norm, trace_norm, mode='full')
    return rho[len(rho) // 2:] / len(trace_norm)

def trazas(model):
    p=[]
    by = n_p.array(model['by'])
    bz = n_p.array(model['bz'])
    wy = n_p.array(model['wy'])
    wz = n_p.array(model['wz'])
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), tight_layout=True)   
    fig.suptitle("Gráficas de Trazas")
    for a in model:
        if a == 'by':
            ax[0,0].set_xlabel('Iteraciones')
            ax[0,0].set_ylabel('Traza')
            ax[0,0].set_title(a)
            ax[0,0].plot(by)
        elif a == 'bz':
            ax[0,1].set_xlabel('Iteraciones')
            ax[0,1].set_ylabel('Traza')
            ax[0,1].set_title(a)
            for i in range(bz.shape[1]):
                ax[0,1].plot(bz[:,i])
        elif a == 'wy':
            ax[0,2].set_xlabel('Iteraciones')
            ax[0,2].set_ylabel('Traza')
            ax[0,2].set_title(a)
            for i in range(wy.shape[1]):
                ax[0,2].plot(wy[:,i,0])
        elif a == 'wz':
            ax[1,0].set_xlabel('Iteraciones')
            ax[1,0].set_ylabel('Traza')
            ax[1,0].set_title('w_z_1')
            for i in range(wz.shape[2]):
                ax[1,0].plot(wz[:,0,i])

            ax[1,1].set_xlabel('Iteraciones')
            ax[1,1].set_ylabel('Traza')
            ax[1,1].set_title('w_z_2')
            for i in range(wz.shape[2]):
                ax[1,1].plot(wz[:,1,i])

def correlaciones(model):    
    p=[]
    by = n_p.array(model['by'])
    bz = n_p.array(model['bz'])
    wy = n_p.array(model['wy'])
    wz = n_p.array(model['wz'])
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), tight_layout=True)   
    fig.suptitle("Gráficas de Correlaciones")
    for a in model:
        if a == 'by':
            ax[0,0].set_xlabel('Retardo')
            ax[0,0].set_ylabel('Traza')
            ax[0,0].set_title(a)
            ax[0,0].plot(autocorrelation(by))
        elif a == 'bz':
            ax[0,1].set_xlabel('Retardo')
            ax[0,1].set_ylabel('Traza')
            ax[0,1].set_title(a)
            for i in range(bz.shape[1]):
                ax[0,1].plot(autocorrelation(bz[:,i]))
        elif a == 'wy':
            ax[0,2].set_xlabel('Retardo')
            ax[0,2].set_ylabel('Traza')
            ax[0,2].set_title(a)
            for i in range(wy.shape[1]):
                ax[0,2].plot(autocorrelation(wy[:,i,0]))
        elif a == 'wz':
            ax[1,0].set_xlabel('Retardo')
            ax[1,0].set_ylabel('Traza')
            ax[1,0].set_title('w_z_1')
            for i in range(wz.shape[2]):
                ax[1,0].plot(autocorrelation(wz[:,0,i]))

            ax[1,1].set_xlabel('Iteraciones')
            ax[1,1].set_ylabel('Traza')
            ax[1,1].set_title('w_z_2')
            for i in range(wz.shape[2]):
                ax[1,1].plot(autocorrelation(wz[:,1,i]))