module Interp
# Code for interpolation for various orders
using LinearAlgebra
using Statistics

export CubicSpline, interp, slope, slope2, pchip, pchip2, pchip3

const EPS = 1e-3 # rel error allowed on extrapolation


"""
    CubicSpline(x, a, b, c, d, alphabar)

Concrete type for the data needed to do a cubic spline interpolation.
"""
struct CubicSpline{R,T,RT}
    x::R
    a::T
    b::T
    c::T
    d::T
    alphabar::RT
end

"""
    PCHIP(x, y, d, h)

Concrete type for the data needed to do a piecewise continuous hermite interpolation.
"""
struct PCHIP{R,T,RT}
    x::R
    y::T
    d::T
    h::RT
end

"""
    CubicSpline(x, y)

Constructor for `CubicSpline` of values `y` evaluated at `x`.
"""
function CubicSpline(x::AbstractVector, y::AbstractVector)
    len = size(x, 1)
    if len < 3
        error("CubicSpline requires at least three points for interpolation")
    end

    # Pre-allocate and fill columns and diagonals
    yy = similar(y)
    dd = similar(x)

    # Scale x so that the alpha values are better
    α = diff(x)
    alphabar = Statistics.mean(α)
    @. α = α/alphabar

    # precompute, saving squares and divisions, but at the cost of allocations
    # the crossover point for precomputing `dusq` versus doing all the divisions in place
    # is for `len` between 100 and 1000
    du = 1 ./ α
    dusq = 1 ./ α.^2

    yy[1] = 3*(y[2] - y[1])*dusq[1]
    dd[1] = 2*du[1]
    for i = 2:len-1
        yy[i] = 3*(y[i+1]*dusq[i] + y[i]*(dusq[i-1] - dusq[i]) - y[i-1]*dusq[i-1])
        dd[i] = 2*(du[i-1] + du[i])
    end
    yy[len] = 3*(y[len] - y[len-1])*dusq[len-1]
    dd[len] = 2*du[len-1]

    # Solve the tridiagonal system for the derivatives D
    dm = Tridiagonal(du,dd,du)
    D = dm\yy

    # fill the arrays of spline coefficients
    a = y[1:len-1]

    # silly but makes the code more transparent
    b = D[1:len-1]

    c = similar(a)
    d = similar(a)
    for i = 1:len-1
        c[i] = 3*(y[i+1] - y[i])*dusq[i] - 2*D[i]*du[i] - D[i+1]*du[i]
        d[i] = 2*(y[i] - y[i+1])*dusq[i]*du[i] + D[i]*dusq[i] + D[i+1]*dusq[i]
    end

    return CubicSpline(x, a, b, c, d, alphabar)
end

function CubicSpline(x::AbstractRange, y::AbstractVector)
    len = length(x)
    if len < 3
        error("CubicSpline requires at least three points for interpolation")
    end

    # Pre-allocate and fill columns and diagonals
    yy = similar(y)
    dl = ones(len-1)
    dd = fill(4.0, len)
    dd[1] = 2.0
    dd[len] = 2.0
    yy[1] = 3*(y[2] - y[1])
    @views @. yy[2:len-1] = 3*(y[3:len] - y[1:len-2])
    yy[len] = 3*(y[len] - y[len-1])

    # Solve the tridiagonal system for the derivatives D
    dm = Tridiagonal(dl,dd,dl)
    D = dm\yy

    # fill the arrays of spline coefficients
    a = y[1:len-1]

    # silly but makes the code more transparent
    b = D[1:len-1]

    @views c = @. 3*(y[2:len] - y[1:len-1]) - 2*D[1:len-1] - D[2:len]
    @views d = @. 2*(y[1:len-1] - y[2:len]) + D[1:len-1] + D[2:len]
    α = step(x)

    return CubicSpline(x, a, b, c, d, α)
end

"""
    pchip(x, y)

Creates the PCHIP structure needed for piecewise continuous cubic spline interpolation.

This function uses the mean value of the slopes between data points on either side of the
interpolation point.
"""
function pchip(x, y)
    len = size(x, 1)
    if len < 3
        error("PCHIP requires at least three points for interpolation")
    end

    h = diff(x)

    d = similar(y)
    d[1] = (y[2] - y[1])/h[1]
    for i = 2:len-1
        d[i] = (y[i+1]/h[i] + y[i]*(1/h[i-1] - 1/h[i]) - y[i-1]/h[i-1])/2
    end
    d[len] = (y[len] - y[len-1])/h[len-1]

    PCHIP(x, y, d, h)
end

"""
    pchip2(x, y)

PCHIP with a quadratic fit to determine slopes.
"""
function pchip2(x, y)
    len = size(x,1)
    if len < 3
        error("PCHIP requires at least three points for interpolation")
    end

    h = diff(x)

    # Pre-allocate and fill columns and diagonals
    d = similar(y)
    d[1] = (y[2] - y[1])/h[1]
    for i = 2:len-1
        d[i] = (y[i] - y[i-1])*h[i]/(h[i-1]*(h[i-1] + h[i])) +
            (y[i+1] - y[i])*h[i-1]/(h[i]*(h[i-1] + h[i]))
    end
    d[len] = (y[len] - y[len-1])/h[len-1]

    PCHIP(x, y, d, h)
end

"""
    pchip3(x, y)

This is the "real" PCHIP. Choose this for the closest match to MATLAB.
"""
function pchip3(x, y)
    len = size(x, 1)

    len < 3 && error("PCHIP requires at least three points for interpolation")
    issorted(x) || error("pchip3: array of x values is not monotonic")

    h = diff(x)

    Δ = diff(y)./h

    # Pre-allocate and fill columns and diagonals
    d = similar(y)

    d[1] = Δ[1]
    for i = 2:len-1
        if Δ[i]*Δ[i-1] < 0
            d[i] = 0
        else
            d[i] = (Δ[i] + Δ[i-1])/2
        end
    end
    d[len] = Δ[len-1]
    for i = 1:len-1
        if Δ[i] == 0
            d[i] = 0
            d[i+1] = 0
        else
            α = d[i]/Δ[i]
            β = d[i+1]/Δ[i]

            l = hypot(α, β)
            if l > 3
                τ = 3/l
                d[i] = τ*α*Δ[i]
                d[i+1] = τ*β*Δ[i]
            end
        end
    end

    PCHIP(x, y, d, h)
end


"""
    interp(cs::CubicSpline, v)
    interp(pc::PCHIP, v)

Interpolate to the value corresonding to v.

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
cs = CubicSpline(x,y)
v = interp(cs, 1.2)
```
"""
function interp(cs::CubicSpline, v)
    # Find v in the array of x's
    if v < minimum(cs.x) || v > maximum(cs.x)
        error("Extrapolation not allowed")
    end
    
    segment = region(cs.x, v)

    t = (v - cs.x[segment])/cs.alphabar
    
    return cs.a[segment] + t*(cs.b[segment] + t*(cs.c[segment] + t*cs.d[segment]))
end
(cs::CubicSpline)(v) = interp(cs, v)

function interp(pc::PCHIP, v)
    if v*(1 + EPS) < minimum(pc.x)
        error("Extrapolation not allowed, $v < $(minimum(pc.x))")
    end
    if v*(1 - EPS) > maximum(pc.x)
        error("Extrapolation not allowed, $v > $(maximum(pc.x))")
    end

    i = region(pc.x, v)

    function aux(t)
        t² = t^2
        t³ = t²*t

        ϕ = 3*t² - 2*t³
        ψ = t³ - t²

        return ϕ, ψ
    end

    t13 = (pc.x[i+1] - v)/pc.h[i]
    t24 = (v - pc.x[i])/pc.h[i]

    ϕ13, ψ13 = aux(t13)
    ϕ24, ψ24 = aux(t24)

    H1, H3 = ϕ13, -pc.h[i]*ψ13
    H2, H4 = ϕ24, pc.h[i]*ψ24

    yv = pc.y[i]*H1 + pc.y[i+1]*H2 + pc.d[i]*H3 + pc.d[i+1]*H4

    # For reasons I have yet to understand completely, this can blow up sometimes.
    # Revert to linear interpolation in this case
    if isnan(yv)
        h = pc.x[i+1] - pc.x[i]
        if h > 0
            t = (pc.x[i+1] - v)/h
        else
            error("interp data is not monotonic, x[i] = $(x[i]), x[i+1]=$(x[i+1])")
        end
        @warn "reverting to linear interpolation"
        yv = pc.y[i] + t*(pc.y[i+1] - pc.y[i])
    end

    return yv
end
(pc::PCHIP)(v) = interp(pc,v)


"""
    slope(cs::CubicSpline, v)

Derivative at the point corresonding to v.

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
cs = CubicSpline(x,y)
v = slope(cs, 1.2)
```
"""
function slope(cs::CubicSpline, v)
    if v < minimum(cs.x) || v > maximum(cs.x)
        error("Extrapolation not allowed")
    end

    segment = region(cs.x, v)
   
    t = (v-cs.x[segment])/cs.alphabar
    
    return cs.b[segment] + t*(2*cs.c[segment] + t*3*cs.d[segment])
end

"""
    slope(pc::PCHIP, v)

Derivative at the point corresponding to v.

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
pc = pchip(x,y)
v = slope(pc, 1.2)
```
"""
function slope(pc::PCHIP, v)
    if v < minimum(pc.x) || v > maximum(pc.x)
        error("Extrapolation not allowed")
    end

    i = region(pc.x, v)

    function aux(t)
        t² = t^2

        ϕp = 6*t - 6*t²
        ψp = 3*t² - 2*t

        return ϕp, ψp
    end

    t13 = (pc.x[i+1] - v)/pc.h[i]
    t24 = (v - pc.x[i])/pc.h[i]

    ϕp13, ψp13 = aux(t13)
    ϕp24, ψp24 = aux(t24)

    H1p, H3p = -ϕp13/pc.h[i], ψp13
    H2p, H4p = ϕp24/pc.h[i], ψp24

    return pc.y[i]*H1p + pc.y[i+1]*H2p + pc.d[i]*H3p + pc.d[i+1]*H4p
end

"""
    slope2(cs::CubicSpline, v)

Second derivative at the point corresponding to v.

# Examples
```
x = cumsum(rand(10))
y = cos.(x);
cs = CubicSpline(x,y)
v = slope2(cs, 1.2)
```
"""
function slope2(cs::CubicSpline, v)
    if v < minimum(cs.x) || v > maximum(cs.x)
        error("Extrapolation not allowed")
    end

    segment = region(cs.x, v)
   
    t = (v-cs.x[segment])/cs.alphabar
    
    return 2*cs.c[segment] + 6*t*cs.d[segment]
end

function region(x::AbstractArray, v)
    # Binary search
    len = size(x,1)
    li = 1
    ui = len
    mi = div(li+ui,2)
    done = false
    while !done
        if v < x[mi]
            ui = mi
            mi = div(li+ui,2)
        elseif v > x[mi+1]
            li = mi
            mi = div(li+ui,2)
        else
            done = true
        end
        if mi == li
            done = true
        end
    end
    return mi
end

function region(x::AbstractRange, y)
    min(floor(Int,(y-first(x))/step(x)), length(x)-2) + 1
end

end # module Interp
