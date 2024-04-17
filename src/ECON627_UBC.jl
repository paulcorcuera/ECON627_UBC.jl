module ECON627_UBC

using Random, Parameters, LinearAlgebra, Optim, ForwardDiff


export ols, Ω, GMM, TSGMM, TSLS, nls, NLGMM, MLE

#This will contain all functions related to estimation and standard error calculation for the UBC course ECON627



########## Linear Models #####################

# Ordinary Least Squares 
function ols(X,Y)
    n = length(Y)

    xx = X'*X
    xy = X'*Y
    θhat = xx\xy
    res = Y - X*θhat


    xr = X.*res
    avar = n*(xx \ (xr' * xr) / xx)

    se = sqrt.(diag(avar))

    return (θhat = θhat , se = se)
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
    θhat=(X'*Z*W*Z'*X)\(X'*Z*W*Z'*Y)
    Γ = Z'*X/n
    omega = Ω(Y-X*θhat,Z)

    avar = (Γ'*W*Γ)\(Γ'*W*omega*W*Γ)/(Γ'*W*Γ)
    se = sqrt.(diag(avar))
    
    return (θhat = θhat , se = se)
end


# Two Step GMM 
function TSGMM(Y,X,Z)
    
    n=length(Y)
    
    # 2SLS
    PZ = Z*( (Z'*Z)\Z' )
    β2SLS = (X'*PZ*X)\(X'*PZ*Y)
    Q = Z'*X/n
    Ω1 = Ω(Y-X*β2SLS,Z)
    
    # Two-step efficient GMM
    WGMM=inv(Ω1);
    θhat=(X'*Z*WGMM*Z'*X)\(X'*Z*WGMM*Z'*Y)
    Ω2=Ω(Y-X*θhat,Z)
    WGMM=inv(Ω2)

    avar = inv(Q'*WGMM*Q)
    se=sqrt.(diag(avar))
    
    return  (θhat = θhat , se = se)
    
end

# Two Stage Least Squares 
function TSLS(Y, X, Z)
    n = length(Y)
    W = inv(Z'*Z)

    θhat =(X'*Z*W*Z'*X)\(X'*Z*W*Z'*Y)
    Γ = Z'*X/n

    omega = Ω(Y-X*θhat,Z)
    
    avar = (Γ'*W*Γ)\(Γ'*W*omega*W*Γ)/(Γ'*W*Γ)
    se = sqrt.(diag(avar))
    
    return (θhat = θhat , se = se)
end

########### Non-linear Models ##################### 

# Non-linear Least Squares 
function nls(f, y, x ) 

    n = size(x, 1)
    # Objective function
    obj = b -> sum((y - f(x, b)) .^ 2) ;
    
    #Initial Value 
    beta0 = zeros(size(x,2))

    # We set the criterion function as an instance that we can differentiate twice
    td = TwiceDifferentiable(obj, beta0 ; autodiff = :forward)
    o = optimize(td, beta0, Newton(), Optim.Options() )

    if !Optim.converged(o)
        error("Minimization failed.")
    end

    θhat = Optim.minimizer(o)


    # Get residuals
    r_hat = y - f(x, θhat)

    # Get asyvar, we compute the gradient of f with respect to θhat
    v = map(i -> ForwardDiff.gradient(z -> f(x[i, :]', z), θhat), 1:n)
    md = vcat(v'...)

    me = md .* r_hat  ; mmd = md' * md
    avar = ( mmd/n) \ (me' * me /n) / (mmd/n)

    se = sqrt.(diag(avar));

    return (θhat = θhat, se = se )
end

# Criterion Function for NonLinear GMM is some function Q(θ,Y,X,Z,W)


# Non-linear GMM 
function NLGMM(Y,X,Z,W,g)
    #g is the moment condition function, this is a FUNCTION
    n = length(Y)
 
    gl = (y,x,z,θ) -> g(y,x,z,θ)/n

    function gvec(Y,X,Z,θ)
        v = map(i -> gl(Y[i],X[i, :]',Z[i,:]',  θ), 1:n)
        v = sum([ v[i] for i in 1:n  ])
        v = [v...]
        return v
    end
    Q = θ -> transpose(gvec(Y,X,Z,θ)) * W * gvec(Y,X,Z,θ)
 
    res=optimize(θ->Q(θ,Y,X,Z,W),[0.0],NewtonTrustRegion(); autodiff = :forward)

    #Initial Value 
    initval = zeros(size(X,2))
    td = TwiceDifferentiable(Q, initval ; autodiff = :forward)
    #Other option is to use trust region
    # res=optimize(θ->Q(θ,Y,X,Z,W),initval,NewtonTrustRegion(); autodiff = :forward)
    res = optimize(td, initval, Newton(), Optim.Options() )
    θhat=Optim.minimizer(res)
    
    #Standard Error
    # To get asyvar, we compute the gradient of g with respect to theta
    # Jacobian will yield dg(W,θ)/dθ' (lxk matrix)
    v = map(i -> ForwardDiff.jacobian(θ -> gl(Y[i],X[i, :]',Z[i,:]', θ), θhat), 1:n)
    dg = sum([ v[i] for i in 1:n ])
   
    v = map(i -> gl(Y[i],X[i, :]',Z[i,:]', θhat), 1:n)
    outerprod = sum([ [v[i]...]*[v[i]...]' for i in 1:n ])
    
    mmd = dg' *W* dg
    avar = (mmd) \ dg' *W*(outerprod)*W*dg / (mmd)

    se = sqrt.(diag(avar))

    
    return (θhat = θhat, se = se )
end


# Maximum Likelihood  
function MLE(Y,X,LogL)
    #LogL is a function of the data, this is a FUNCTION of b, Y, X 
    initval = zeros(size(X,2))

    res=optimize(b->LogL(b,X,Y),initval,LBFGS();autodiff=:forward)
    θhat=Optim.minimizer(res)


    avar = ForwardDiff.hessian(b -> LogL(b, X,Y), θhat )
    avar = inv(avar)
    se =  sqrt.(diag(avar))

    return (θhat = θhat, se = se )
end

# Minimum Distance 
#function MD(π, Y, X, Z,g,W)
    # π is the dependent variaθhatle you want to fit 
    # g is a function of the data, this is a FUNCTION
#    Q = (π-g(Y,X,Z))'*W*(π-g(Y,X,Z))

    #Minimizer
#    res=optimize(θ->Q(θ,Y,X,Z,W),[0.0],NewtonTrustRegion(); autodiff = :forward)
#    θhathat=Optim.minimizer(res)
    
    # Get asyvar, we compute the gradient of f with respect to θhat
#    v = map(i -> ForwardDiff.gradient(θ->Q(θ,Y,X,Z,W), θhathat), 1:n)
#    md = vcat(v'...)

#    me = md .* r_hat; mmd = md' * md
#    avar = mmd \ (me' * me) / mmd

#    se = sqrt.(diag(avar));

    #Standard Error
#    avar=ForwardDiff.hessian(θ->Q(θ,Y,X,Z,W),θhat)
#    se=sqrt.( diag(inv(avar))/n)

#    return (θhat = θhathat, se = se )
#end

end # module
