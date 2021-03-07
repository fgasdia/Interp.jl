using Test
try
    using Interp
catch
    push!(LOAD_PATH, pwd())
    using Interp
end
import Interp.region

const unitTests = true
const graphicsTests = false
const bumpTests = false

if graphicsTests || bumpTests
    using Plots
end

function regular_tests()
    @testset "regular interpolation" begin
        # Test not enough points exception
        x = range(1.0, stop=2.0, length=2)
        y = [2.0, 4.0]
        @test_throws ErrorException CubicSpline(x,y)

        x = range(1.0, stop=3.25, length=4)
        y = [1.5, 3.0, 3.7, 2.5]
        cs = CubicSpline(x,y)
        @test_throws ErrorException interp(cs, 0.0)
        @test_throws ErrorException interp(cs, 4.0)

        # Check region
        @test region(x, 1.0) == 1
        @test region(x, 1.2) == 1
        @test region(x, 3.25) == 3
        @test region(x, 2.0) == 2
        @test region(x, 2.8) == 3

        # Check spline at knots
        @test interp(cs, 1.0) == 1.5
        @test interp(cs, 1.75) == 3.0
        @test isapprox(interp(cs, 3.25), 2.5; atol=1e-14)

        # Check spline with unit spacing of knots
        len = 5
        x = range(0.0, stop=4.0, length=len)
        y = sin.(x)
        cs = CubicSpline(x,y)
        dy = cos.(x)
        for i = 1:len-1
            @test cs.a[i] == y[i]
            @test isapprox(cs.a[i] + cs.b[i] + cs.c[i] + cs.d[i], y[i+1]; atol=1.e-12)
            @test isapprox(cs.b[i], dy[i]; atol=0.08)
        end
        for i = 1:len-2
            @test isapprox(cs.b[i] + 2*cs.c[i] + 3*cs.d[i], dy[i+1]; atol=0.08)
        end

        # Check second derivatives at end points
        @test isapprox(0, cs.c[1]; atol=1e-14);
        @test isapprox(0, cs.c[len-1] + 3*cs.d[len-1]; atol=1e-14);
    end
end

function irr_coef_tests()
    @testset "irregular interpolation coefficients test" begin
        x = [0.2, 1.4, 3.8, 5.7]
        y = [1.5, 3.0, 3.7, 2.5]
        n = length(x)
        csi = CubicSpline(x,y)
        alpha = (x[2:n] - x[1:n-1])/csi.alphabar
        for i = 1:length(x)-1
            ap = csi.a[i] + csi.b[i]*alpha[i] + csi.c[i]*alpha[i]^2 + csi.d[i]*alpha[i]^3
            @test isapprox(ap, y[i+1])
        end
        for i = 1:length(x)-2
            bp = csi.b[i] + 2*csi.c[i]*alpha[i] + 3*csi.d[i]*alpha[i]^2
            @test isapprox(bp, csi.b[i+1])
        end
    end
end

function irregular_tests()
    @testset "irregular interpolation" begin
        # Test not enough points exception
        x = [1.0, 2.0]
        y = [2.0, 4.0]
        @test_throws ErrorException CubicSpline(x,y)

        x = [0.2, 1.4, 3.8, 5.7]
        y = [1.5, 3.0, 3.7, 2.5]
        csi = CubicSpline(x,y)
        @test_throws ErrorException interp(csi, 0.0)
        @test_throws ErrorException interp(csi, 6.0)

        # Check region
        @test region(x, 0.3) == 1
        @test region(x, 0.2) == 1
        @test region(x, 5.7) == 3
        @test region(x, 2.1) == 2
        @test region(x, 4.0) == 3

        # Check spline at knots
        @test interp(csi, 0.2) == 1.5
        @test interp(csi, 1.4) == 3.0
        @test isapprox(interp(csi, 5.7), 2.5; atol=1e-14)

        # Check spline with unit spacing of knots
        x = range(0.,4.,step=1.)
        y = sin.(x)
        cs = CubicSpline(x,y)
        csi = CubicSpline(collect(x),y)
        for i = 1:4
            @test csi.a[i] == cs.a[i]
            @test csi.b[i] == cs.b[i]
            @test csi.c[i] == cs.c[i]
            @test csi.d[i] == cs.d[i]
            @test csi.a[i] == y[i]
            @test isapprox(csi.a[i] + csi.b[i] + csi.c[i] + csi.d[i], y[i+1]; atol=1e-12)
        end

        # Check meeting knot conditions
        for i = 1:3
            di = csi.b[i+1]
            dip = csi.b[i] + 2*csi.c[i] + 3*csi.d[i]
            @test isapprox(di, dip; atol=1e-12)
        end
        for i = 1:3
            ddi = 2*csi.c[i+1]
            ddip = 2*csi.c[i] + 6*csi.d[i]
            @test isapprox(ddi, ddip; atol=1e-12)
        end

        # Second derivatives at end points
        @test isapprox(csi.c[1], 0.0, atol=1e-12)
        @test isapprox(2*csi.c[4]+6*csi.d[4], 0.0; atol=1e-12)

        # Test matching boundary conditions with unequally spaced knots
        x = [0.0, 0.7, 2.3, 3.0, 4.1]
        y = sin.(x)
        csi = CubicSpline(x,y)
        for i = 1:4
            @test csi.a[i] == y[i]
            alpha = x[i+1]-x[i]
            yend = csi.a[i] + csi.b[i]*alpha + csi.c[i]*alpha^2 + csi.d[i]*alpha^3
            # @test isapprox(yend, y[i+1], atol=1.e-12)
        end

        # Check for continuity near knot 2
        eps = 0.0001
        vl = x[2] - eps
        vg = x[2] + eps
        yl = interp(csi, vl)
        yg = interp(csi, vg)
        @test abs(yl-yg) < 2*eps
        sl = slope(csi, vl)
        sg = slope(csi, vg)
        @test abs(sl-sg) < 2*eps
        sl2 = slope2(csi, vl)
        sg2 = slope2(csi, vg)
        @test abs(sl2-sg2) < 2*eps

        # Check meeting knot conditions
        for i = 1:3
            alpha = (x[i+1]-x[i])/csi.alphabar
            dip = csi.b[i+1]
            di = csi.b[i]+2*csi.c[i]*alpha+3*csi.d[i]*alpha^2
            @test isapprox(di, dip; atol=1e-12)
        end
        for i = 1:3
            alpha = (x[i+1]-x[i])/csi.alphabar
            ddi = 2*csi.c[i+1]
            ddip = 2*csi.c[i] + 6*csi.d[i]*alpha
            @test isapprox(ddi, ddip; atol=1e-12)
        end

        # Second derivatives at end points
        @test isapprox(csi.c[1], 0.0; atol=1e-12)
        alpha = (x[5] - x[4])/csi.alphabar
        @test isapprox(2*csi.c[4] + 6*csi.d[4]*alpha, 0.0; atol=1e-12)
    end
end

function graphics_tests()
    x = range(0.0, stop=pi, length=10)
    y = sin.(x)

    cs = CubicSpline(x,y)
    xx = range(0.0, stop=pi, length=97)
    yy = [interp(cs,v) for v in xx]
    yyy = sin.(xx)

    
    scatter(x,y, markershape=:circle, label="data", title="Regular Interpolation")
    plot!(xx,yy, linestyle=:dash, label="cubic")
    plot!(xx,yyy, linestyle=:dot, label="exact")
    
    x = cumsum(rand(10));
    x = (x.-x[1]).*pi/(x[10].-x[1])
    y = sin.(x)
    cs = CubicSpline(x,y)
    xx = range(0.0, stop=pi, length=97)
    yy = [interp(cs,v) for v in xx]
    yyy = sin.(xx)

    scatter(x,y, markershape=:circle, label="data", title="Irregular Interpolation, 10 Points")
    plot!(xx,yy,  label="cubic")
    plot!(xx,yyy,linestyle=:dot, label="exact")
end

function bump_tests()
    x = [0.0, 0.1, 0.2, 0.3, 0.35, 0.55, 0.65, 0.75];
    y = [0.0, 0.01, 0.02, 0.03, 0.5, 0.51, 0.52, 0.53];
    xx = range(0.0,stop=0.75,length=400);
    sp = CubicSpline(x,y);
    yy = [interp(sp, v) for v in xx]
    pc = pchip(x,y)
    yyy = [interp(pc,v) for v in xx]
    pc2 = pchip2(x,y)
    yyy2 = [interp(pc2,v) for v in xx]

    scatter(x,y, markershape=:circle, label="data", title="Cubic Interpolation")
    plot!(xx,yy, linestyle=:dash, label="spline")
    plot!(xx,yyy, linestyle=:dash, label="mean")
    plot!(xx,yyy2, linestyle=:dash, label="quad")

    pc3 = pchip3(x,y)
    yyy3 = [interp(pc3,v) for v in xx]
    scatter(x, y, markershape=:circle, label="data", title="PCHIP Interpolation")
    plot!(xx, yyy3, linestyle=:dash, label="PCHIP")
end

function regular_pchip_tests()
    @testset "Regular pchip" begin
    end;
end

function irregular_pchip_tests()
    x = [1.0, 1.8, 2.5, 3.0, 3.9]
    y = cos.(x)
    pc = pchip(x,y)

    @testset "Irregular pchip" begin
        for i=1:5
            # Continuity
            @test interp(pc,x[i]) == y[i]
        end
        for i = 2:4
            # Continuity of slope
            eps = 0.000001
            @test isapprox(slope(pc,x[i]-eps), slope(pc,x[i]+eps); atol=4*eps)
        end
    end
end

if unitTests
    regular_tests()
    irr_coef_tests()
    irregular_tests()
    # regular_pchip_tests()
    # irregular_pchip_tests()
end

if graphicsTests
    graphics_tests()
end

if bumpTests
    bump_tests()
end