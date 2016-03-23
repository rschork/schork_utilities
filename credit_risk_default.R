calcPosteriorForProportion <- function(successes, total, a, b)
{
  # Adapted from triplot() in the LearnBayes package
  # Plot the prior, likelihood and posterior:
  likelihood_a = successes + 1; likelihood_b = total - successes + 1
  posterior_a = a + successes;  posterior_b = b + total - successes
  theta = seq(0.005, 0.995, length = 500)
  prior = dbeta(theta, a, b)
  likelihood = dbeta(theta, likelihood_a, likelihood_b)
  posterior  = dbeta(theta, posterior_a, posterior_b)
  m = max(c(prior, likelihood, posterior))
  plot(theta, posterior, type = "l", ylab = "Density", lty = 2, lwd = 3,
       main = paste("beta(", a, ",", b, ") prior, B(", total, ",", successes, ") data,",
                    "beta(", posterior_a, ",", posterior_b, ") posterior"), ylim = c(0, m), col = "red")
  lines(theta, likelihood, lty = 1, lwd = 3, col = "blue")
  lines(theta, prior, lty = 3, lwd = 3, col = "green")
  legend(x=0.8,y=m, c("Prior", "Likelihood", "Posterior"), lty = c(3, 1, 2),
         lwd = c(3, 3, 3), col = c("green", "blue", "red"))
  # Print out summary statistics for the prior, likelihood and posterior:
  calcBetaMode <- function(aa, bb) { BetaMode <- (aa - 1)/(aa + bb - 2); return(BetaMode); }
  calcBetaMean <- function(aa, bb) { BetaMean <- (aa)/(aa + bb); return(BetaMean); }
  calcBetaSd   <- function(aa, bb) { BetaSd <- sqrt((aa * bb)/(((aa + bb)^2) * (aa + bb + 1))); return(BetaSd); }
  prior_mode      <- calcBetaMode(a, b)
  likelihood_mode <- calcBetaMode(likelihood_a, likelihood_b)
  posterior_mode  <- calcBetaMode(posterior_a, posterior_b)
  prior_mean      <- calcBetaMean(a, b)
  likelihood_mean <- calcBetaMean(likelihood_a, likelihood_b)
  posterior_mean  <- calcBetaMean(posterior_a, posterior_b)
  prior_sd        <- calcBetaSd(a, b)
  likelihood_sd   <- calcBetaSd(likelihood_a, likelihood_b)
  posterior_sd    <- calcBetaSd(posterior_a, posterior_b)
  print(paste("mode for prior=",prior_mode,", for likelihood=",likelihood_mode,", for posterior=",posterior_mode))
  print(paste("mean for prior=",prior_mean,", for likelihood=",likelihood_mean,", for posterior=",posterior_mean))
  print(paste("sd for prior=",prior_sd,", for likelihood=",likelihood_sd,", for posterior=",posterior_sd))
}

betabinlike=function(alpha,beta){-sum(dbetabinom.ab(def,n,alpha,beta,log=TRUE))}
n = array(12)
def = array(0)
est = betabinlike(1,1,0,12)

calcPosteriorForProportion(0,12,1,1)


x=c(-0.99,-1.00,0.05,-2.29,-1.31,-0.73,1.06,0.75,-1.61,1.21,-0.01,-0.13,-0.44,0.19,0.55,-1.59,-1.61,1.20,-0.97,-0.46,1.22,-0.96,1.00,-0.29,0.00,0.62,0.78,1.84,1.53,1.58)

y=c(-0.37,-1.77,-1.33,-1.62,-2.06,1.42,0.45,1.65,-0.91,3.38,-0.22,-0.42,0.11,0.79,-0.06,-1.05,-1.29,-0.41,-1.79,-0.45,0.91,-0.65,-1.21,-0.78,-0.45,0.78,2.38,2.36,1.81,0.75)
m <- cbind(x, y) 
cor(m, method="kendall", use="pairwise") 


# We start by defining the quantities whose value is known.

S0=11; # Equity at time 0, i.e. today.
sigmaS=0.7; # Instantaneous volatility of equity
r=0.06; # Risk-free rate on the market
T=1; # Maturity
B=15; # Face value of debt obligation, i.e. liabilities.

# We then need to write down the function we will minimize in order to obtain 
# V0, the value of company’s assets today, and sigmaV, assets’ volatility

# To write a function in R, we use the command “function”. Refer to the R intro
# for more details.

Merton_solve=function(parm){
  
  V0=parm[1] #initial value for V0
  sigmaV=parm[2] #initial value for sigmaV
  
  # And now, all the quantities we have seen in the slides.
  
  d1=(log(V0/B)+(r+sigmaV^2/2)*T)/(sigmaV*sqrt(T)) 
  d2=d1-sigmaV*sqrt(T)
  F=V0*pnorm(d1)-B*exp(-r*T)*pnorm(d2)-S0
  G=pnorm(d1)*sigmaV*V0-sigmaS*S0
  
  #  Finally the result of our function:  
  return(F^2+G^2)
}

# For the minimization step, in order to find V_0 and sigma_V,
# we need to specify two initial values.

# Let us choose V_0=13, and sigmaV=0.5.
# Other plausible values can obviously be chosen.

solutions=optim(c(V0=13,sigmaV=0.5),Merton_solve)

# What are the estimated values?
V0=solutions$par[1]
sigmaV=solutions$par[2]

# Let us compute d1 and d2 explicitly:

d1=(log(27.3625/15)+(0.06+0.3^2/2)*1)/(0.3*sqrt(1))
d2=d1-0.3*sqrt(1)


# And finally the probability of default in one year for our company

pnorm(-d2)

p = .03+.13*exp(-35*.005)
WCDR = pnorm((qnorm(0.005)+sqrt(p)*qnorm(0.999))/sqrt(1-p))
RWA=12.5*50000000*0.2*(WCDR-0.005)*1
RWA/1000000
.08*RWA/1000000