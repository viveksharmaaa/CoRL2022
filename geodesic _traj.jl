function _chebpts(N)
    K = N-1
    n = (cos.(π*(K:-1:0)/K) .+ 1)./2
    w = similar(n)

    Kh = K % 2 == 0 ? K^2 - 1 : K^2
    w[1] = w[end] = 0.5/Kh

    Kt = div(K, 2)
    for k = 1:Kt
        wk = 0.0
        for j = 0:Kt
            β = (j == 0) || (j == K/2) ? 1 : 2
            wk += β/K/(1-4*j^2)*cos(2*π*j*k/K)
        end
        w[K-k+1] = w[k+1] = wk
    end

    return n, w
end

function _chebpoly(nodes, D)
    # D : Max degree

    N = length(nodes)
    T = zeros(D, N)

    for i = 1:D
        for j = 1:N
            if i == 1
                T[i,j] = 1
            elseif i == 2
                T[i,j] = nodes[j]
            else
                T[i,j] = 2*nodes[j]*T[i-1,j] - T[i-2,j]
            end
        end
    end

    return T
end

function _chebpolyder(T, nodes, D)
    # D : Max degree

    N = length(nodes)
    dT = similar(T)

    for i = 1:D
        for j = 1:N
            if i == 1
                dT[i,j] = 0
            elseif i == 2
                dT[i,j] = 1
            else
                dT[i,j] =  2*T[i-1,j] + 2*nodes[j]*dT[i-1,j] - dT[i-2,j]
            end
        end
    end

    return dT
end

function coeffs(γ, x0, xf, A, T)
    Ti = @view T[:,2:end-1]
    z = A*[2*Ti*γ'; x0'; xf']
    (@view z[1:end-2,:])'
end

function energy(c, T, Ts, weights, W)
    E = 0.0
    x = c*T
    γs = c*Ts
    @inbounds for k = 1:length(weights)
        xv = view(x,:,k)
        γsv = view(γs,:,k)
        E += γsv'*(W(xv)\γsv)*weights[k]
    end
    return E
end

function energy(γ, x0, xf, A, T, Ts, weights, W)
    c = coeffs(γ, x0, xf, A, T)
    energy(c, T, Ts, weights, W)
end

#Change num_dim_x, X_MIN, X_MAX, vars, xf index in Wf(x) for each case

using ForwardDiff
using LinearAlgebra
using Optim
using PyCall
using NPZ
using Random
import LineSearches: BackTracking

np = pyimport("numpy")
num_dim_x = 4
vars = npzread("/home/vivek/PycharmProjects/CoRL2022/SEGWAY.npz")
traj = npzread("/home/vivek/PycharmProjects/CoRL2022/SEGWAY_closed_pts.npz")

function Wf(x)
        xf = reshape(x[2:4],(1,3))
        W = reshape(tanh.(xf*(vars["arr_0"]') + vars["arr_1"]') * vars["arr_2"]',(num_dim_x,num_dim_x))'
	#W = reshape((tanh.(xf*(vars["arr_0"]') + reshape(vars["arr_1"],(1,128))))* vars["arr_2"]',(4,4))
	W = W'*W
end 


N = 7
D = 5

nodes, weights = _chebpts(N)
T = _chebpoly(nodes, D)
Ts = _chebpolyder(T, nodes, D)
Ti = T[:,2:N-1]; Te = T[:,[1,N]]
A = inv([2*Ti*Ti' Te;Te' zeros(2,2)])
num_points = size(traj["arr_0"],1)

xd = np.zeros((num_dim_x,num_points))
xc = np.zeros((num_dim_x,num_points))
RME= np.zeros((1,num_points))

for i = 1:num_points
        if i == 1
	   l = 0
        end
	xs =  traj["arr_0"][i,:]
	x =   traj["arr_1"][i,:]
        print("$(i):")
	printstyled("xstar $(xs)\n";color=1)
	printstyled("x $(x)\n";color=2)
	γ0 = zeros(length(xs), N-2)
	γ(s) = xs*(1-s) + x*s #initialize with straight line
	for j = 2:N-1
 	    γ0[:,j-1] = γ(nodes[j])
	end
	obj(γ) = energy(γ, xs, ForwardDiff.value.(x), A, T, Ts, weights, Wf)
	try
           ret = optimize(obj, γ0, BFGS(linesearch=BackTracking()); autodiff=:forward)
           l = l + 1
           ret = optimize(obj, γ0, BFGS(linesearch=BackTracking()); autodiff=:forward)
           γ0 .= Optim.minimizer(ret)
	   E = minimum(ret)
	   printstyled("RE $(E)\n";color=3)
	   xd[:,l] = xs
           xc[:,l] = x
           RME[1,l] = E
        catch e
           println("Raise Exception")
	   continue
        end
end
npzwrite("SEGWAY_closed.npz",Dict("xd" => xd, "x" => xc, "RE" => RME))
print("DONE")




