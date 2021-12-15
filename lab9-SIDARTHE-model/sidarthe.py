
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt



def after_dayX(alpha, beta, gamma, delta, epsilon, theta, zeta, eta, mu, nu, tau, lambda_, kappa, xi, pho, sigma):
	# Transmittion rate (Typically, α > γ)
	alpha_val 	= alpha # α
	beta_val 	= beta  # β
	gamma_val 	= gamma # γ
	delta_val 	= delta # δ
	# Probability rate of detection (Typically, θ > ε)
	epsilon_val = epsilon # ε
	theta_val 	= theta # θ
	# Probability rate at which an infected subject
	zeta_val 	= zeta # ζ
	eta_val 	= eta  # η
	# Rate at which undetected and detected infected subjects
	mu_val 		= mu # µ
	nu_val		= nu # ν
	# Mortality rate
	tau_val 	= tau # τ
	# Rate of recovery
	lambda_val  = lambda_ # λ
	kappa_val 	= kappa # κ
	xi_val  	= xi # ξ
	pho_val 	= pho # ρ
	sigma_val	= sigma # σ
	return (alpha_val, beta_val, gamma_val, delta_val, epsilon_val, theta_val, zeta_val, eta_val, mu_val, nu_val, tau_val, lambda_val, kappa_val, xi_val, pho_val, sigma_val)



def model():
	m = GEKKO()
	m.time = np.linspace(0, 350, num=500)

	e_init = 0.0
	h_init = 0.0
	t_init = 0.0
	r_init = 2/60e+6
	a_init = 1/60e+6
	d_init = 20/60e+6
	i_init = 200/60e+6
	s_init = 1 - (i_init + d_init + a_init + r_init + t_init + h_init + e_init)

	day = int(input("Simulate after day X = "))

	if day == 1:
		# After day (α    , β    , γ    , δ    , ε    , θ    , ζ    , η    , µ    , ν    , τ   , λ    , κ    , ξ    , ρ    , σ    )
		alpha, beta, gamma, delta, epsilon, theta, zeta, eta, mu, nu, tau, lambda_, kappa, xi, pho, sigma \
		= after_dayX(0.570, 0.011, 0.456, 0.011, 0.171, 0.371, 0.125, 0.125, 0.017, 0.027, 0.01, 0.034, 0.017, 0.017, 0.034, 0.017)
	elif day == 4:
		# After day 4
		alpha, beta, gamma, delta, epsilon, theta, zeta, eta, mu, nu, tau, lambda_, kappa, xi, pho, sigma \
		= after_dayX(0.422, 0.0057, 0.285, 0.0057, 0.171, 0.371, 0.125, 0.125, 0.017, 0.027, 0.01, 0.034, 0.017, 0.017, 0.034, 0.017)
	elif day == 12:
		# After day 12
		alpha, beta, gamma, delta, epsilon, theta, zeta, eta, mu, nu, tau, lambda_, kappa, xi, pho, sigma \
		= after_dayX(0.422, 0.0057, 0.285, 0.0057, 0.143, 0.371, 0.125, 0.125, 0.017, 0.027, 0.01, 0.034, 0.017, 0.017, 0.034, 0.017)
	elif day == 22:
		# After day 22
		alpha, beta, gamma, delta, epsilon, theta, zeta, eta, mu, nu, tau, lambda_, kappa, xi, pho, sigma \
		= after_dayX(0.360, 0.005, 0.200, 0.005, 0.143, 0.371, 0.034, 0.034, 0.008, 0.015, 0.01, 0.08, 0.017, 0.017, 0.017, 0.017)
	else:
		print("Enter another number")
		exit()
	
	s,i,d,a,r,t,h,e = m.Array(m.Var, 8)

	s.value = s_init
	i.value = i_init
	d.value = d_init
	a.value = a_init
	r.value = r_init
	t.value = t_init
	h.value = h_init
	e.value = e_init
	
	# Modeling for COVID patients and recovering it
	m.Equations([s.dt() == -(s * (alpha*i + beta*d + gamma*a + delta*r))])
	m.Equations([i.dt() == (s * (alpha*i + beta*d + gamma*a + delta*r)) - (epsilon + zeta + lambda_) * i])
	m.Equations([d.dt() == (epsilon*i - (eta + pho) * d)])
	m.Equations([a.dt() == (zeta*i - (theta + mu + kappa) * a)])
	m.Equations([r.dt() == (eta*d + theta*a -(nu + zeta) * r)])
	m.Equations([t.dt() == (mu*a + nu*r - (sigma + tau) * t)])
	m.Equations([h.dt() == (lambda_*i + pho*d + kappa*a + zeta*r + sigma*t)])
	m.Equations([e.dt() == (tau*t)])
	
	m.options.IMODE = 7 # 7: Dynamic Sequential Simulation

	m.solve(disp=False) # Unable for displaying results

	plt.plot(m.time, s.value, label="Susceptible")
	plt.plot(m.time, i.value, label="Infected")
	plt.plot(m.time, d.value, label="Diagnosed")
	plt.plot(m.time, a.value, label="Ailing")
	plt.plot(m.time, r.value, label="Recognized")
	plt.plot(m.time, t.value, label="Threatened")
	plt.plot(m.time, h.value, label="Healed")
	plt.plot(m.time, e.value, label="Extinct")

	plt.title("SIDARTHE Model in COVID-19 treatment")
	plt.xlabel("Time")
	plt.ylabel("Fraction")

	plt.grid()
	plt.legend()
	plt.show()



if __name__ == "__main__":
	model()
