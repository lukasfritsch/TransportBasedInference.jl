using TransportBasedInference

QuadratureSparse = gausshermite_weights(3, 2; sparse=true)
QuadratureTensor = gausshermite_weights(3, 2; sparse=false)

## Plot quadrature points
refrange = range(-2, 2, 100)
lognormal = zeros(Nlog, Nlog)

for (i,x) in enumerate(refrange)
    for (j,y) in enumerate(refrange)
        lognormal[i,j] = logpdf(MvNormal(zeros(2), 1.0), [x;y])
    end
end

plt_points = plot(size=(800,800))
contour!(plt_points, refrange, refrange, lognormal,
    colormap=:ice, colorbar=false)
scatter!(plt_points, QuadratureTensor.points[1,:], QuadratureTensor.points[2,:],
    label="Tensor", markersize=10, alpha=0.6)
scatter!(plt_points, QuadratureSparse.points[1,:], QuadratureSparse.points[2,:],
    label="Sparse", markersize=5, alpha=0.8)
