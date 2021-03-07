module Interp
# Code for interpolation for various orders
using LinearAlgebra
using Statistics
import Base.length

export CubicSpline, interp, slope, slope2, pchip, pchip2, pchip3

const eps = 1e-3 # rel error allowed on extrapolation

"""
    CubicSpline(x,a,b,c,d)

concrete type for holding the data needed
    to do a cubic spline interpolation
"""
abstract type AbstractSpline end

struct CubicSpline <: AbstractSpline
    x::Union{Array{Float64,1},
        StepRangeLen{Float64,
            Base.TwicePrecision{Float64},
            Base.TwicePrecision{Float64}}}
    a::Array{Float64,1}
    b::Array{Float64,1}
    c::Array{Float64,1}
    d::Array{Float64,1}
    alphabar::Float64
end

struct ComplexSpline <: AbstractSpline
    x::Union{Array{Float64,1},
        StepRangeLen{Float64,
            Base.TwicePrecision{Float64},
            Base.TwicePrecision{Float64}}}
    a::Array{Complex{Float64},1}
    b::Array{Complex{Float64},1}
    c::Array{Complex{Float64},1}
    d::Array{Complex{Float64},1}
    alphabar::Float64
end

"""
    PCHIP(x,a,b,c,d)

concrete type for holding the data needed
    to do a piecewise continuous hermite interpolation
"""
struct PCHIP
    x::Union{Array{Float64,1},
        StepRangeLen{Float64,
            Base.TwicePrecision{Float64},
            Base.TwicePrecision{Float64}}}
    y::Array{Float64,1}
    d::Array{Float64,1}
    h::Array{Float64,1}
end

"""
    CubicSpline(x,y)

Creates the CubicSpline structure needed for cubic spline
interpolation

# Arguments
- `x`: an array of x values at which the function is known
- `y`: an array of y values corresponding to these x values
"""
function CubicSpline(x::Array{Float64,1}, y::Array{Float64,1})
    len = size(x,1)
    if len<3
        error("CubicSpline requires at least three points for interpolation")
    end
    # Pre-allocate and fill columns and diagonals
    yy = zeros(typeof(x[1]),len)
    du = zeros(typeof(x[1]),len-1)
    dd = zeros(typeof(x[1]),len)
    # Scale x so that the alpha values are better
    alpha = x[2:len].-x[1:len-1]
    alphabar = Statistics.mean(alpha)
    alpha = alpha/alphabar
    yy[1] = 3*(y[2]-y[1])/alpha[1]^2
    du = 1 ./alpha
    dd[1] = 2/alpha[1]
    yy[2:len-1] = 3*(y[3:len]./alpha[2:len-1].^2
        .+y[2:len-1].*(alpha[1:len-2].^(-2).-alpha[2:len-1].^(-2))
        .-y[1:len-2]./alpha[1:len-2].^2)
    dd[2:len-1] = 2*(1 ./alpha[1:len-2] .+ 1 ./alpha[2:len-1])
    yy[len] = 3*(y[len]-y[len-1])/alpha[len-1]^2
    dd[len] = 2/alpha[len-1]
    # Solve the tridiagonal system for the derivatives D
    dm = Tridiagonal(du,dd,du)
    D = dm\yy
    # fill the arrays of spline coefficients
    a = y[1:len-1]
    # silly but makes the code more transparent
    b = D[1:len-1]
    # ditto
    c = 3 .*(y[2:len].-y[1:len-1])./alpha[1:len-1].^2 .-
        2*D[1:len-1]./alpha[1:len-1].-D[2:len]./alpha[1:len-1]
    d = 2 .*(y[1:len-1].-y[2:len])./alpha[1:len-1].^3 .+
        D[1:len-1]./alpha[1:len-1].^2 .+
        D[2:len]./alpha[1:len-1].^2
    CubicSpline(x, a, b, c, d, alphabar)
end


function CubicSpline(x::Array{Float64,1}, y::Array{Complex{Float64},1})
    len = size(x,1)
    if len<3
        error("CubicSpline requires at least three points for interpolation")
    end
    # Pre-allocate and fill columns and diagonals
    yy = zeros(typeof(y[1]),len)
    du = zeros(typeof(x[1]),len-1)
    dd = zeros(typeof(x[1]),len)
    alpha = x[2:len].-x[1:len-1]
    alphabar = Statistics.mean(alpha)
    alpha = alpha/alphabar
    yy[1] = 3*(y[2]-y[1])/alpha[1]^2
    du = 1 ./alpha
    dd[1] = 2/alpha[1]
    yy[2:len-1] = 3*(y[3:len]./alpha[2:len-1].^2
        .+y[2:len-1].*(alpha[1:len-2].^(-2).-alpha[2:len-1].^(-2))
        .-y[1:len-2]./alpha[1:len-2].^2)
    dd[2:len-1] = 2*(1 ./alpha[1:len-2] .+ 1 ./alpha[2:len-1])
    yy[len] = 3*(y[len]-y[len-1])/alpha[len-1]^2
    dd[len] = 2/alpha[len-1]
    # Solve the tridiagonal system for the derivatives D
    dm = Tridiagonal(du,dd,du)
    D = dm\yy
    # fill the arrays of spline coefficients
    a = y[1:len-1]
    # silly but makes the code more transparent
    b = D[1:len-1]
    # ditto
    c = 3 .*(y[2:len].-y[1:len-1])./alpha[1:len-1].^2 .-
        2*D[1:len-1]./alpha[1:len-1].-D[2:len]./alpha[1:len-1]
    d = 2 .*(y[1:len-1].-y[2:len])./alpha[1:len-1].^3 .+
        D[1:len-1]./alpha[1:len-1].^2 .+
        D[2:len]./alpha[1:len-1].^2
    ComplexSpline(x, a, b, c, d, alphabar)
end

function CubicSpline(x::StepRangeLen{Float64,Base.TwicePrecision{Float64},
    Base.TwicePrecision{Float64}}, y::Array{Float64,1})
    len = length(x)
    if len<3
        error("CubicSpline requires at least three points for interpolation")
    end
    # Pre-allocate and fill columns and diagonals
    yy = zeros(len)
    dl = ones(len-1)
    dd = 4.0 .* ones(len)
    dd[1] = 2.0
    dd[len] = 2.0
    yy[1] = 3*(y[2]-y[1])
    yy[2:len-1] = 3*(y[3:len].-y[1:len-2])
    yy[len] = 3*(y[len]-y[len-1])
    # Solve the tridiagonal system for the derivatives D
    dm = Tridiagonal(dl,dd,dl)
    D = dm\yy
    # fill the arrays of spline coefficients
    a = y[1:len-1]
    # silly but makes the code more transparent
    b = D[1:len-1]
    # ditto
    c = 3 .*(y[2:len].-y[1:len-1]).-2*D[1:len-1].-D[2:len]
    d = 2 .*(y[1:len-1].-y[2:len]).+D[1:len-1].+D[2:len]
    alpha = step(x);
    CubicSpline(x, a, b, c, d, alpha)
end


function CubicSpline(x::StepRangeLen{Float64,Base.TwicePrecision{Float64},
    Base.TwicePrecision{Float64}}, y::Array{Complex{Float64},1})
    len = length(x)
    if len<3
        error("CubicSpline requires at least three points for interpolation")
    end
    # Pre-allocate and fill columns and diagonals
    yy = zeros(Complex{Float64}, len)
    dl = ones(len-1)
    dd = 4.0 .* ones(len)
    dd[1] = 2.0
    dd[len] = 2.0
    yy[1] = 3*(y[2]-y[1])
    yy[2:len-1] = 3*(y[3:len].-y[1:len-2])
    yy[len] = 3*(y[len]-y[len-1])
    # Solve the tridiagonal system for the derivatives D
    dm = Tridiagonal(dl,dd,dl)
    D = dm\yy
    # fill the arrays of spline coefficients
    a = y[1:len-1]
    # silly but makes the code more transparent
    b = D[1:len-1]
    # ditto
    c = 3 .*(y[2:len].-y[1:len-1]).-2*D[1:len-1].-D[2:len]
    d = 2 .*(y[1:len-1].-y[2:len]).+D[1:len-1].+D[2:len]
    alpha = x.step;
    ComplexSpline(x, a, b, c, d)
end


# This version of pchip uses the mean value of the slopes
# between data points on either side of the interpolation point
"""
    pchip(x,y)

Creates the PCHIP structure needed for piecewise
    continuous cubic spline interpolation

# Arguments
- `x`: an array of x values at which the function is known
- `y`: an array of y values corresonding to these x values
"""
function pchip(x::Array{Float64,1}, y::Array{Float64,1})
    len = size(x,1)
    if len<3
        error("PCHIP requires at least three points for interpolation")
    end
    h = x[2:len].-x[1:len-1]
    # Pre-allocate and fill columns and diagonals
    d = zeros(len)
    d[1] = (y[2]-y[1])/h[1]
    for i=2:len-1
        d[i] = (y[i+1]/h[i]+y[i]*(1/h[i-1]-1/h[i])-y[i-1]/h[i-1])/2
    end
    d[len] = (y[len]-y[len-1])/h[len-1]
    PCHIP(x,y,d,h)
end

# PCHIP with quadratic fit to determine slopes
function pchip2(x::Array{Float64,1}, y::Array{Float64,1})
    len = size(x,1)
    if len<3
        error("PCHIP requires at least three points for interpolation")
    end
    h = x[2:len].-x[1:len-1]
    # Pre-allocate and fill columns and diagonals
    d = zeros(len)
    d[1] = (y[2]-y[1])/h[1]
    for i=2:len-1
        d[i] = (y[i]-y[i-1])*h[i]/(h[i-1]*(h[i-1]+h[i])) +
            (y[i+1]-y[i])*h[i-1]/(h[i]*(h[i-1]+h[i]))
    end
    d[len] = (y[len]-y[len-1])/h[len-1]
    PCHIP(x,y,d,h)
end


# Real PCHIP
function pchip3(x::Array{Float64,1}, y::Array{Float64,1})
    len = size(x,1)
    if len<3
        error("PCHIP requires at least three points for interpolation")
    end
    for i = 2:length(x)
        if x[i] <= x[i-1]
            error("pchip3: array of x values is not monotonic at x = $(x[i+1])")
        end
    end
    h = x[2:len].-x[1:len-1]
    # test for monotonicty
    del = (y[2:len].-y[1:len-1])./h
    # Pre-allocate and fill columns and diagonals
    d = zeros(len)
    d[1] = del[1]
    for i=2:len-1
        if del[i]*del[i-1] < 0
            d[i] = 0
        else
            d[i] = (del[i]+del[i-1])/2
        end
    end
    d[len] = del[len-1]
    for i=1:len-1
        if del[i] == 0
            d[i] = 0
            d[i+1] = 0
        else
            alpha = d[i]/del[i]
            beta = d[i+1]/del[i]
            if alpha^2+beta^2 > 9
                tau = 3/sqrt(alpha^2+beta^2)
                d[i] = tau*alpha*del[i]
                d[i+1] = tau*beta*del[i]
            end
        end
    end
    PCHIP(x,y,d,h)
end


"""
    interp(cs::CubicSpline, v::Float)

Interpolate to the value corresonding to v

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
cs = CubicSpline(x,y)
v = interp(cs, 1.2)
```
"""
function interp(cs::AbstractSpline, v::Float64)
    # Find v in the array of x's
    if (v<cs.x[1]) | (v>cs.x[length(cs.x)])
        error("Extrapolation not allowed")
    end
    segment = region(cs.x, v)
    if cs.x isa StepRangeLen
        # regularly spaced points
        t = (v-cs.x[segment])/step(cs.x)
    else
        # irregularly spaced points
        t = (v-cs.x[segment])/cs.alphabar
    end
    cs.a[segment] + t*(cs.b[segment] + t*(cs.c[segment] + t*cs.d[segment]))
end
# alias
(cs::AbstractSpline)(v::Float64) = interp(cs,v)



function interp(pc::PCHIP, v::Float64)
    if v*(1+eps)<first(pc.x)
        error("Extrapolation not allowed, $v<$(first(pc.x))")
    end
    if v*(1-eps)>last(pc.x)
        error("Extrapolation not allowed, $v>$(last(pc.x))")
    end
    i = region(pc.x, v)
    phi(t) = 3*t^2 - 2*t^3
    psi(t) = t^3 - t^2
    H1(x) = phi((pc.x[i+1]-v)/pc.h[i])
    H2(x) = phi((v-pc.x[i])/pc.h[i])
    H3(x) = -pc.h[i]*psi((pc.x[i+1]-v)/pc.h[i])
    H4(x) = pc.h[i]*psi((v-pc.x[i])/pc.h[i])
    yv = pc.y[i]*H1(v) + pc.y[i+1]*H2(v) + pc.d[i]*H3(v) + pc.d[i+1]*H4(v)
    # For reasons I have yet to understand completely, this can blow
    # up sometimes. Revert to linear interpolation in this case
    if isnan(yv)
        h = pc.x[i+1] - pc.x[i]
        if h > 0
            t = (pc.x[i+1]-v)/h
        else
            error("interp data is not monotonic, x[i] = $(x[i]), x[i+1]=$(x[i+1])")
        end
        println("warning, reverting to linear interpolation")
        yv = pc.y[i] + t*(pc.y[i+1]-pc.y[i])
    end
    yv
end
#alias
(pc::PCHIP)(v::Float64) = interp(pc,v)


"""
    slope(cs::CubicSpline, v::Float)

Derivative at the point corresonding to v

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
cs = CubicSpline(x,y)
v = slope(cs, 1.2)
```
"""
function slope(cs::CubicSpline, v::Float64)
    # Find v in the array of x's
    if (v<cs.x[1]) | (v>cs.x[length(cs.x)])
        error("Extrapolation not allowed")
    end
    segment = region(cs.x, v)
    if cs.x isa StepRangeLen
        # regularly spaced points
        t = (v-cs.x[segment])/step(cs.x)
    else
        # irregularly spaced points
        t = (v-cs.x[segment])/cs.alphabar
    end
    cs.b[segment] + t*(2*cs.c[segment] + t*3*cs.d[segment])
end


"""
    slope(pc::PCHIP, v::Float)

Derivative at the point corresponding to v

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
pc = pchip(x,y)
v = slope(pc, 1.2)
```
"""
function slope(pc::PCHIP, v::Float64)
    # Find v in the array of x's
    if (v<pc.x[1]) | (v>pc.x[length(pc.x)])
        error("Extrapolation not allowed")
    end
    i = region(pc.x, v)
    phip(t) = 6*t - 6*t^2
    psip(t) = 3*t^2 - 2*t
    H1p(x) = -phip((pc.x[i+1]-v)/pc.h[i])/pc.h[i]
    H2p(x) = phip((v-pc.x[i])/pc.h[i])/pc.h[i]
    H3p(x) = psip((pc.x[i+1]-v)/pc.h[i])
    H4p(x) = psip((v-pc.x[i])/pc.h[i])
    pc.y[i]*H1p(v) + pc.y[i+1]*H2p(v) + pc.d[i]*H3p(v) + pc.d[i+1]*H4p(v)
end


"""
    slope2(cs::CubicSpline, v::Float)

Second derivative at the point corresponding to v

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
cs = CubicSpline(x,y)
v = slope2(cs, 1.2)
```
"""
function slope2(cs::CubicSpline, v::Float64)
    # Find v in the array of x's
    if (v<cs.x[1]) | (v>cs.x[length(cs.x)])
        error("Extrapolation not allowed")
    end
    segment = region(cs.x, v)
    if cs.x isa StepRangeLen
        # regularly spaced points
        t = (v-cs.x[segment])/step(cs.x)
    else
        # irregularly spaced points
        t = (v-cs.x[segment])/cs.alphabar
    end
    2*cs.c[segment] + 6*t*cs.d[segment]
end

function region(x::AbstractArray, v::Float64)
    # Binary search
    len = size(x,1)
    li = 1
    ui = len
    mi = div(li+ui,2)
    done = false
    while !done
        if v<x[mi]
            ui = mi
            mi = div(li+ui,2)
        elseif v>x[mi+1]
            li = mi
            mi = div(li+ui,2)
        else
            done = true
        end
        if mi == li
            done = true
        end
    end
    mi
end

function region(x::StepRangeLen{Float64,Base.TwicePrecision{Float64},
    Base.TwicePrecision{Float64}}, y::Float64)
    min(floor(Int,(y-first(x))/step(x)),length(x)-2) + 1
end

end # module Interp
