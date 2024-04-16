module ECON627_UBC

using Random, Parameters, LinearAlgebra, Optim, ForwardDiff


export ols, Ω, GMM, TSGMM, TSLS, nls, NLGMM, MLE

#This will contain all functions related to estimation and standard error calculation for the UBC course ECON627



########## Linear Models #####################

# Ordinary Least Squares 
function ols(X,Y)
    n = length(Y)

    b = (X'*X)\(X'*Y)
    res = Y - X*b
    Xres = X.*res
    avar = (X'*X)\(Xres'*Xres)/(X'*X)

    se = sqrt.(diag(avar)/n)

    return (b = b , se = se)
end

# Avar for GMM 
function Ω(U,Z)
    n=length(U)
    zr = Z.*U
    omega = (zr' * zr)/n
    
    return omega
end

# Linear GMM 
function GMM(W, Y, X, Z)
    n = length(Y)
    b=(X'*Z*W*Z'*X)\(X'*Z*W*Z'*Y)
    Γ = Z'*X/n
    omega = Ω(Y-X*b,Z)

    avar = (Γ'*W*Γ)\(Γ'*W*omega*W*Γ)/(Γ'*W*Γ)
    se = sqrt.(diag(avar)/n)
    
    return (b = b , se = se)
end


# Two Step GMM 
function TSGMM(Y,X,Z)
    
    n=length(Y)
    
    # 2SLS
    PZ = Z*( (Z'*Z)\Z' )
    β2SLS = (X'*PZ*X)\(X'*PZ*Y)
    Q = Z'*X/n
    Ω1 = Ω(Y-β2SLS*X,Z)
    
    # Two-step efficient GMM
    WGMM=inv(Ω1);
    b=(X'*Z*WGMM*Z'*X)\(X'*Z*WGMM*Z'*Y)
    Ω2=Ω(Y-b*X,Z)
    WGMM=inv(Ω2)

    avar = inv(Q'*WGMM*Q)/n
    se=sqrt.(avar)
    
    return  (b = b , se = se)
    
end

# Two Stage Least Squares 
function TSLS(Y, X, Z)
    n = length(Y)
    W = inv(Z'*Z)

    b =(X'*Z*W*Z'*X)\(X'*Z*W*Z'*Y)
    Γ = Z'*X/n
    
    avar = (Γ'*W*Γ)\(Γ'*W*omega*W*Γ)/(Γ'*W*Γ)
    se = sqrt.(diag(avar)/n)
    
    return (b = b , se = se)
end

########### Non-linear Models ##################### 

# Non-linear Least Squares 
function nls(f, y, x ) 

    n = size(x, 1)
    # Objective function
    obj = b -> sum((y - f(x, b)) .^ 2);
    
    #Initial Value 
    beta0 = zeros(size(x,2))

    # We set the criterion function as an instance that we can differentiate twice
    td = TwiceDifferentiable(obj, beta0 ; autodiff = :forward)
    o = optimize(td, beta0, Newton(), Optim.Options() )

    if !Optim.converged(o)
        error("Minimization failed.")
    end

    bhat = Optim.minimizer(o)


    # Get residuals
    r_hat = y - f(x, bhat)

    # Get asyvar, we compute the gradient of f with respect to b
    v = map(i -> ForwardDiff.gradient(z -> f(x[i, :]', z), bhat), 1:n)
    md = vcat(v'...)

    me = md .* r_hat; mmd = md' * md
    avar = mmd \ (me' * me) / mmd

    se = sqrt.(diag(avar));

    return (b = bhat, se = se )
end

# Criterion Function for NonLinear GMM is some function Q(θ,Y,X,Z,W)


# Non-linear GMM 
function NLGMM(Y,X,Z,Q)
    #Q is the criterion function, this is a FUNCTION
    n = length(Y)
    #first step GMM
    res=optimize(θ->Q(θ,Y,X,Z,W),[0.0],NewtonTrustRegion(); autodiff = :forward)
    b=Optim.minimizer(res)
    
    #Standard Error
    avar=ForwardDiff.hessian(θ->Q(θ,Y,X,Z,W),b)
    se=sqrt.( diag(inv(avar))/n)
    
    return (b = b, se = se )
end


# Maximum Likelihood  
function MLE(Y,X,LogL)
    #LogL is a function of the data, this is a FUNCTION

    res=optimize(b->LogL(b,Y,X),[0.0;0.0;],LBFGS();autodiff=:forward)
    bhat=Optim.minimizer(res)


    avar = ForwardDiff.hessian(b -> LogL(b, Y, X), bhat )
    se =  sqrt.(diag(inv(avar)))

    return (b = bhat, se = se )
end

# Minimum Distance 
#function MD(π, Y, X, Z,g,W)
    # π is the dependent variable you want to fit 
    # g is a function of the data, this is a FUNCTION
#    Q = (π-g(Y,X,Z))'*W*(π-g(Y,X,Z))

    #Minimizer
#    res=optimize(θ->Q(θ,Y,X,Z,W),[0.0],NewtonTrustRegion(); autodiff = :forward)
#    bhat=Optim.minimizer(res)
    
    # Get asyvar, we compute the gradient of f with respect to b
#    v = map(i -> ForwardDiff.gradient(θ->Q(θ,Y,X,Z,W), bhat), 1:n)
#    md = vcat(v'...)

#    me = md .* r_hat; mmd = md' * md
#    avar = mmd \ (me' * me) / mmd

#    se = sqrt.(diag(avar));

    #Standard Error
#    avar=ForwardDiff.hessian(θ->Q(θ,Y,X,Z,W),b)
#    se=sqrt.( diag(inv(avar))/n)

#    return (b = bhat, se = se )
#end

end # module
