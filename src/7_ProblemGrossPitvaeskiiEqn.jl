#Fairly sure this function is now mistake free!!!

function FFEM2ErvGrd(ffemgridfile)
    # Open the file
    fin = open(ffemgridfile, "r")

    # Read the first line and extract nVert, ntri, nbdy
    buffer = readline(fin)
    n1 = parse.(Int, split(buffer))
    nVert, ntri, nbdy = n1[1], n1[2], n1[3]

    # Initialize arrays
    nodeco = zeros(nVert, 2)
    elnode = zeros(Int,ntri, 3)
    bdynde = zeros(Int, nbdy)

    # Read vertex coordinates
    for ii in 1:nVert
        buffer = readline(fin)
        vert = parse.(Float64, split(buffer))
        nodeco[ii, 1] = vert[1]
        nodeco[ii, 2] = vert[2]
    end

    # Read element nodes
    for ii in 1:ntri
        buffer = readline(fin)
        n1 = parse.(Int, split(buffer))
        elnode[ii, 1] = n1[1]
        elnode[ii, 2] = n1[2]
        elnode[ii, 3] = n1[3]
    end

    # Initialize boundary segment arrays
    bdymat = zeros(Int, nbdy, 3)  # Ensure it's a 2D array
    nedgept = zeros(Int, nbdy)

    # Read boundary segments and detect edge changes
    buffer = readline(fin)
    n1 = parse.(Int, split(buffer))
    bdymat[1, :] .= n1  # Use broadcasting to assign the row
    nedgept[1] = 1
    edgeind = n1[3]
    edgetot = 1

    for ii in 2:nbdy
        buffer = readline(fin)
        n1 = parse.(Int, split(buffer))
        bdymat[ii, :] .= n1  # Ensure the row is assigned correctly

        if n1[3] != edgeind
            # New edge detected
            nedgept[2 * edgetot] = ii - 1
            edgetot += 1
            nedgept[2 * edgetot - 1] = ii
            edgeind = n1[3]
        end
    end

    nedgept[2 * edgetot] = nbdy

    # Order the boundary segments
    bdyord = zeros(Int, 0, 3)  # Ensure it's a 2D array
    for ii in 1:edgetot
        ifnd = 0
        ij = 1
        while ij <= edgetot && ifnd == 0
            if ii == bdymat[nedgept[2 * ij - 1], 3]
                bdyord = vcat(bdyord, bdymat[nedgept[2 * ij - 1]:nedgept[2 * ij], :])
                ifnd = 1
            end
            ij += 1
        end
    end

    bdymat = bdyord  # Rebuild bdymat with ordered segments

    # Handle the first edge
    bdynde[1:nedgept[2]] .= bdymat[1:nedgept[2], 1]

    # Handle the rest of the edges
    for ii in 2:edgetot
        ndenxt = bdymat[nedgept[2 * (ii - 1)], 2]

        if bdymat[nedgept[2 * ii - 1], 1] == ndenxt
            # Boundary segment is listed consistently
            bdynde[nedgept[2 * ii - 1]:nedgept[2 * ii]] .= bdymat[nedgept[2 * ii - 1]:nedgept[2 * ii], 1]
        else
            # Segment is listed backwards
            bdynde[nedgept[2 * ii - 1]:nedgept[2 * ii]] .= reverse(bdymat[nedgept[2 * ii - 1]:nedgept[2 * ii], 1])
        end
    end

    # Close the file
    close(fin)

    return nodeco, elnode, bdynde, nVert
end

###Also fairly sure this works!
function midEDGEgen(nodeco, elnode, bdynde)
    ntri = size(elnode, 1)  # Number of triangles
    nVert = size(nodeco, 1)  # Number of vertices
    nnds = nVert  # Start counting nodes from the current number of vertices
    
    elnode = hcat(elnode, zeros(Int, size(elnode, 1), 3))

    # Initialize edgeset and edge count
    edgeset = spzeros(Int64, nVert, nVert)
    nedge = 0

    # Define the vertex ordering for edges of the triangle
    vec1 = [2, 3, 1, 2]  # Triangle vertices indices
    
    # Loop over each triangle
    for itrg in 1:ntri
        for iedge in 1:3
            n1 = elnode[itrg, vec1[iedge]]
            n2 = elnode[itrg, vec1[iedge + 1]]
            nmin = min(n1, n2)
            nmax = max(n1, n2)
            
            # Check if this edge already exists
            if edgeset[nmin, nmax] == 0
                # Create new mid-edge node
                nedge += 1
                nnds += 1
                new_node = (nodeco[n1, :] + nodeco[n2, :]) / 2.0  # Compute the midpoint
                nodeco = vcat(nodeco, new_node')  # Append the new row (ensure it’s a row vector)
                edgeset[nmin, nmax] = nedge  # Store edge in the edgeset
                edgenode = nedge
            else
                edgenode = edgeset[nmin, nmax]
            end

            # Assign the new edge node to the element
            elnode[itrg, 3 + iedge] = edgenode
        end
    end

    # Now to introduce the boundary edge node list
    nbdy = size(bdynde, 1)
    bdyedge = []

    for iby in 1:(nbdy - 1)
        n1 = bdynde[iby, 1]
        n2 = bdynde[iby + 1, 1]
        nmin = min(n1, n2)
        nmax = max(n1, n2)
        vbc = min(bdynde[iby, 2], bdynde[iby + 1, 2])
        sbc = min(bdynde[iby, 4], bdynde[iby + 1, 4])
        push!(bdyedge, [edgeset[nmin, nmax], vbc, -1, sbc])  # Add boundary edge info
    end
    
    # Handle the last boundary condition (closing the boundary)
    n1 = bdynde[1, 1]
    n2 = bdynde[nbdy, 1]
    nmin = min(n1, n2)
    nmax = max(n1, n2)
    vbc = min(bdynde[1, 2], bdynde[nbdy, 2])
    sbc = min(bdynde[1, 4], bdynde[nbdy, 4])
    push!(bdyedge, [edgeset[nmin, nmax], vbc, -1, sbc])  # Add the final boundary edge

    return nodeco, elnode, bdynde, bdyedge, nVert, nedge
end

#Checked Nearly Perfect.... Just unsure about transposes, etc.

function quad_168()
    # Quadrature points for degree 8 exactness
    quad_pts = [
        0.33333333333333   0.33333333333333;
        0.08141482341455   0.45929258829272;
        0.45929258829272   0.45929258829272;
        0.45929258829272   0.08141482341455;
        0.65886138449648   0.17056930775176;
        0.17056930775176   0.17056930775176;
        0.17056930775176   0.65886138449648;
        0.89890554336594   0.05054722831703;
        0.05054722831703   0.05054722831703;
        0.05054722831703   0.89890554336594;
        0.00839477740996   0.26311282963464;
        0.26311282963464   0.72849239295540;
        0.72849239295540   0.00839477740996;
        0.26311282963464   0.00839477740996;
        0.72849239295540   0.26311282963464;
        0.00839477740996   0.72849239295540
    ]
    
    # Quadrature weights
    quad_wghts = [
        0.14431560767779;
        0.09509163426728;
        0.09509163426728;
        0.09509163426728;
        0.10321737053472;
        0.10321737053472;
        0.10321737053472;
        0.03245849762320;
        0.03245849762320;
        0.03245849762320;
        0.02723031417444;
        0.02723031417444;
        0.02723031417444;
        0.02723031417444;
        0.02723031417444;
        0.02723031417444
    ]'
    
    return quad_pts, quad_wghts
end

quad_pts, quad_wghts = quad_168()

### Fairly Sure this is working!
function CtsQuad(quad_pts)
    # Computes continuous quadratic basis functions and their gradients
    nqpt = size(quad_pts, 1)

    # Preallocate arrays for values and gradients
    CQuadVal = zeros(6, nqpt)
    GradCQuadVal = zeros(6, 2, nqpt)

    # Quadrature points
    ξ = quad_pts[:, 1]   # Column vector
    η = quad_pts[:, 2]   # Column vector

    # Basis function values
    CQuadVal[1, :] .= (1.0 .- ξ .- η) .* (1.0 .- 2 .* ξ .- 2 .* η)
    CQuadVal[2, :] .= ξ .* (2 .* ξ .- 1)
    CQuadVal[3, :] .= η .* (2 .* η .- 1)
    CQuadVal[4, :] .= 4 .* ξ .* η
    CQuadVal[5, :] .= 4 .* η .* (1.0 .- ξ .- η)
    CQuadVal[6, :] .= 4 .* ξ .* (1.0 .- ξ .- η)

    # Gradients of basis functions
    GradCQuadVal[1, 1, :] .= -3.0 .+ 4 .* ξ .+ 4 .* η
    GradCQuadVal[2, 1, :] .= 4 .* ξ .- 1
    GradCQuadVal[3, 1, :] .= 0.0
    GradCQuadVal[4, 1, :] .= 4 .* η
    GradCQuadVal[5, 1, :] .= -4 .* η
    GradCQuadVal[6, 1, :] .= 4 .* (1.0 .- 2 .* ξ .- η)

    GradCQuadVal[1, 2, :] .= -3.0 .+ 4 .* ξ .+ 4 .* η
    GradCQuadVal[2, 2, :] .= 0.0
    GradCQuadVal[3, 2, :] .= 4 .* η .- 1
    GradCQuadVal[4, 2, :] .= 4 .* ξ
    GradCQuadVal[5, 2, :] .= 4 .* (1.0 .- ξ .- 2 .* η)
    GradCQuadVal[6, 2, :] .= -4 .* ξ

    return CQuadVal, GradCQuadVal
end

function initGPE(xy_pts)
    # Separate x and y coordinates
    xpts = xy_pts[:, 1]  # Column vector
    ypts = xy_pts[:, 2]  # Column vector
    npts = size(xy_pts, 1)

    # Initialize velocity as a 2 x npts matrix
    velocity = zeros(2, npts)

    # Constants
    gx = 1
    gy = 1

    # Compute the first row of velocity
    velocity[1, :] .= (gx * gy)^(1/4) * sqrt(π) .* exp.(-0.5 .* (gx .* xpts.^2 .+ gy .* ypts.^2))

    return velocity
end


### Fairly sure also works!

function GPESetup(nodeco, elnode, bdynde,nVert)
    # Boundary setup
    nbdy = size(bdynde, 1)
    bdynde = hcat(bdynde , zeros(Int,nbdy , 1), -1*ones(Int,nbdy , 2)) ;
    nodeco, elnode, bdynde, bdyedge, nVert, nedge = midEDGEgen(nodeco, elnode, bdynde)

    ntri = size(elnode, 1)
    DomPoint = [1]
    TrgMeNxt = [2:ntri; -1] |> vec

    nDom = length(DomPoint)

    # Setup for CtsQuad
    GlobalV = zeros(nVert + nedge)
    NVU = length(GlobalV)

    # Initialize solution vector
    tempV = initGPE(nodeco)
    GlobalV[1:(nVert + nedge)] .= tempV[1, :]

    # Preallocate storage for basis values and gradients
    basisValues = zeros(Float64, 6, 16, ntri)
    gradBasisValues = zeros(Float64, 6, 2, 16, ntri)
    detJglobal = zeros(Float64, ntri)
    quad_wghtsglobal = zeros(Float64, ntri, 16)  # Assuming 16 quadrature points

    for triag_no in 1:ntri
        # Description of triangle
        cotri = nodeco[elnode[triag_no, 1:3], :]

        Jmat = [cotri[2, 1] - cotri[1, 1] cotri[3, 1] - cotri[1, 1];
                cotri[2, 2] - cotri[1, 2] cotri[3, 2] - cotri[1, 2]]
        detJ = abs(Jmat[1, 1] * Jmat[2, 2] - Jmat[1, 2] * Jmat[2, 1])
        JInv = inv(Jmat)

        detJglobal[triag_no] = detJ

        # Evaluate quadrature points and weights
        quad_pts, quad_wghts = quad_168()  # Replace with actual quadrature function
        nqpts = size(quad_pts, 1)

        # Adjust points and weights for true triangle
        xy_pts = (Jmat * quad_pts')'
        xy_pts[:, 1] .+= cotri[1, 1]
        xy_pts[:, 2] .+= cotri[1, 2]
        quad_wghts .*= detJ

        quad_wghtsglobal[triag_no, 1:nqpts] .= quad_wghts'

        # Evaluate basis functions and gradients at quadrature points
        ten1a, Gradten1a = CtsQuad(quad_pts)  # Replace with actual function
        basisValues[:, :, triag_no] .= ten1a

        for iq in 1:nqpts
            gradBasisValues[:, :, iq, triag_no] .= Gradten1a[:, :, iq] * JInv
        end
    end

    # Boundary conditions
    dirbdynde = Int[]
    bdrydof = Int[]
    for i in 1:size(nodeco, 1)
        x, y = nodeco[i, :]
        if abs(y) > 7.999 || abs(x) > 7.999
            push!(dirbdynde, i)
            push!(bdrydof, i)
        end
    end

    return nodeco, elnode, bdynde, GlobalV, NVU, basisValues, gradBasisValues, detJglobal, quad_wghtsglobal, dirbdynde, bdrydof
end


### This should work!!!

function inner_prod_Grad_ten1_Grad_ten1_test(
    nodeco,
    elnode,
    triag_no::Int,
    ten1a::Array{Float64, 2},
    Gradtrue::Array{Float64, 3}
)
    # Description of triangle
    cotri = nodeco[elnode[triag_no, 1:3], :]

    Jmat = [
        cotri[2, 1] - cotri[1, 1] cotri[3, 1] - cotri[1, 1];
        cotri[2, 2] - cotri[1, 2] cotri[3, 2] - cotri[1, 2]
    ]
    detJ = abs(Jmat[1, 1] * Jmat[2, 2] - Jmat[1, 2] * Jmat[2, 1])
    JInv = inv(Jmat)

    # Quadrature points and weights
    quad_pts, quad_wghts = quad_168()
    nqpts = size(quad_pts, 1)

    # Adjust points and weights for true triangle
    xy_pts = (Jmat * quad_pts')'
    xy_pts[:, 1] .+= cotri[1, 1]
    xy_pts[:, 2] .+= cotri[1, 2]
    quad_wghts .= detJ .* quad_wghts

    # Evaluate basis function sizes
    nbas1a = size(ten1a, 1)
    nbas1b = size(ten1a, 1)  # Assuming ten1a and ten1b are the same size

    # Adjust gradients using quadrature weights
    Gradtrue1 = deepcopy(Gradtrue)
    for iq in 1:nqpts
        Gradtrue1[:, :, iq] .= quad_wghts[iq] .* Gradtrue1[:, :, iq]
    end

    # Compute local matrix
    tempM1 = Array{Float64}(undef, nbas1a, nqpts)
    tempM2 = Array{Float64}(undef, nbas1b, nqpts)

    mat1 = zeros(nbas1a, nbas1b)
    mat2 = zeros(nbas1a, nbas1b)

    for k = 1:2  # x and y components
        for iq = 1:nqpts
            tempM1[:, iq] = Gradtrue1[:, k, iq]
            tempM2[:, iq] = Gradtrue[:, k, iq]
        end
        mat_k = tempM1 * tempM2'
        if k == 1
            mat1 .= mat_k
        else
            mat2 .= mat_k
        end
    end

    # Combine x and y contributions
    mat_sum = mat1 + mat2
    localmat = [
        mat_sum zeros(nbas1a, nbas1b);
        zeros(nbas1a, nbas1b) mat_sum
    ]

    # Permute rows and columns for consistency with global numbering
    rdim, cdim = size(localmat)
    
    # Convert StepRange to array of indices using collect()
    rowperm = vcat([collect(ir:nbas1a:rdim) for ir in 1:nbas1a]...)
    colperm = vcat([collect(ic:nbas1b:cdim) for ic in 1:nbas1b]...)

    return localmat[rowperm, colperm]
end


#### This function also works!!!
function inner_prod_ten1_ten1(
    nodeco,
    elnode,
    triag_no::Int,
    ten1a::Array{Float64, 2},
    Gradtrue::Array{Float64, 3}
)
    
# Description of the triangle
cotri = zeros(3, 2)  # Preallocate
cotri[:, 1] = nodeco[elnode[triag_no, 1:3], 1]
cotri[:, 2] = nodeco[elnode[triag_no, 1:3], 2]

Jmat = [
    cotri[2, 1] - cotri[1, 1]  cotri[3, 1] - cotri[1, 1];
    cotri[2, 2] - cotri[1, 2]  cotri[3, 2] - cotri[1, 2]
]
detJ = abs(Jmat[1, 1] * Jmat[2, 2] - Jmat[1, 2] * Jmat[2, 1])
JInv = inv(Jmat)

# Evaluation of quadrature points and quadrature weights
quad_pts, quad_wghts = quad_168()
nqpts = size(quad_pts, 1)

# Adjust points and weights to account for the size of the true triangle
xy_pts = (Jmat * quad_pts')'
xy_pts[:, 1] .+= cotri[1, 1]
xy_pts[:, 2] .+= cotri[1, 2]
quad_wghts .= detJ .* quad_wghts

# Evaluate basis functions and their gradients at quadrature points
nbas1a = size(ten1a, 1)
ten1b = copy(ten1a)  # Assuming ten1a and ten1b are the same size
nbas1b = size(ten1b, 1)

# Adjust ten1b using quadrature weights
for iq in 1:nqpts
    ten1b[:, iq] .= quad_wghts[iq] .* ten1b[:, iq]
end

# Compute the integral evaluations
mat1 = ten1a * ten1b'

# Create the local matrix
localmat = [
    mat1          zeros(size(mat1));
    zeros(size(mat1))  mat1
]
    
# Get the dimensions of the matrix
rdim, cdim = size(localmat)

# Generate row and column permutations
rowperm = vcat([collect(ir:nbas1a:rdim) for ir in 1:nbas1a]...)
colperm = vcat([collect(ic:nbas1b:cdim) for ic in 1:nbas1b]...)

# Reorder the matrix using the permutations
localmat = localmat[rowperm, colperm]

    return localmat
end


### This function also now works!!!

function createMassStiffness(elnode, nodeco, basisValues, gradBasisValues, NVU, nVert)
    Sdim = NVU

    # Preallocate for sparse matrix construction
    i2 = zeros(Int, 100 * Sdim)
    j2 = zeros(Int, 100 * Sdim)
    s2 = zeros(Float64, 100 * Sdim)
    m2 = zeros(Float64, 100 * Sdim)
    number2 = 0

    for itrg in 1:size(elnode, 1)
        # Map local indices to global velocity vector indices
        Vstart = vcat(2 * (elnode[itrg, 1:3] .- 1) .+ 1, 2 * (elnode[itrg, 4:6] .+ nVert .- 1) .+ 1)
        GlTrgVe = reshape([Vstart Vstart .+ 1]',12,1)
        
        ten1a = basisValues[:, :, itrg]
        Gradtrue = gradBasisValues[:, :, :, itrg]

        # Compute local stiffness matrix
        matA11_a = inner_prod_Grad_ten1_Grad_ten1_test(nodeco,elnode,itrg, ten1a, Gradtrue)

        # Compute local mass matrix
        LMass2 = inner_prod_ten1_ten1(nodeco,elnode,itrg, ten1a, Gradtrue)

        # Assemble into global sparse structure
        for count in 1:size(GlTrgVe, 1)
            for count1 in 1:size(GlTrgVe, 1)
                number2 += 1
                if number2 > length(i2)
                    # Dynamically resize arrays if needed
                    i2 = vcat(i2, zeros(Int, length(i2)))
                    j2 = vcat(j2, zeros(Int, length(j2)))
                    s2 = vcat(s2, zeros(Float64, length(s2)))
                    m2 = vcat(m2, zeros(Float64, length(m2)))
                end
                i2[number2] = GlTrgVe[count]
                j2[number2] = GlTrgVe[count1]
                s2[number2] = matA11_a[count, count1]
                m2[number2] = LMass2[count, count1]
            end
        end
    end

    # Construct sparse matrices
    StiffnessMatrix = sparse(i2[1:number2], j2[1:number2], s2[1:number2], 2 * NVU, 2 * NVU)
    MassMatrix = sparse(i2[1:number2], j2[1:number2], m2[1:number2], 2 * NVU, 2 * NVU)

    # Reduce matrices for scalar fields
    StiffnessMatrix = StiffnessMatrix[1:2:end, 1:2:end]
    MassMatrix = MassMatrix[1:2:end, 1:2:end]

    return StiffnessMatrix, MassMatrix
end


function VVfunc(x, y; gx = 1, gy = 1)
    return 0.5 * (gx^2 * x.^2 + gy^2 * y.^2) + 4 * exp(-( (x - 1).^2 + y.^2 ))
end

### Also finally works!!!
function inner_prod_ten1_ten1_scal(
    nodeco, elnode,
    triag_no::Int,
    scalvals,
    ten1a::Array{Float64, 2},
    quad_wghts::Array{Float64, 1}
)

    # Description of triangle
    cotri = nodeco[elnode[triag_no, 1:3], :]

    # Quadrature weights and points
    nqpts = length(quad_wghts)
    nbas1a = size(ten1a,1) ;
    ten1b=copy(ten1a);
    nbas1b = size(ten1a,1) ;

    # Apply scaling to ten1b
    for iq in 1:nqpts
        ten1b[:, iq] .= quad_wghts[iq] * ten1b[:, iq] * scalvals[iq]
    end

    # Compute the local matrix
    mat1 = ten1a * ten1b'

    # Create the 2x2 block matrix
    localmat = [
        mat1  zeros(nbas1a, nbas1a);
        zeros(nbas1a, nbas1a) mat1
    ]

    # Reorder the matrix to match global numbering
    rdim, cdim = size(localmat)
    rowperm = vcat([ir:nbas1a:rdim for ir in 1:nbas1a]...)
    colperm = vcat([ic:nbas1b:cdim for ic in 1:nbas1b]...)

    return localmat[rowperm, colperm]
end


## Also seems to work!!!

function computeMassMatrixGPE(elnode, nodeco, GlobalV, basisValues, NVU, nVert; bet = 10000, VV = VVfunc)
    Sdim = NVU

    # Preallocate for sparse matrix construction
    i2 = zeros(Int, 100 * Sdim)
    j2 = zeros(Int, 100 * Sdim)
    m1 = zeros(Float64, 100 * Sdim)
    number2 = 0

    # GlobalV manipulation
    GlobalVtemp = zeros(Float64, 2 * NVU)
    GlobalVtemp[1:2:end] .= GlobalV
    GlobalVtemp[2:2:end] .= GlobalV
    GlobalV2 = copy(GlobalV)
    GlobalV = GlobalVtemp

    for itrg in 1:size(elnode, 1)
        # Assemble global indices for current element
        Vstart = vcat(2 * (elnode[itrg, 1:3] .- 1) .+ 1, 2 * (elnode[itrg, 4:6] .+ nVert .- 1) .+ 1)
        GlTrgVe = reshape([Vstart Vstart .+ 1]',12,1)

        ten1a = basisValues[:, :, itrg]

        # Coordinates of current triangle
        cotri = nodeco[elnode[itrg, 1:3], :]

        Jmat = [cotri[2, 1] - cotri[1, 1] cotri[3, 1] - cotri[1, 1];
                cotri[2, 2] - cotri[1, 2] cotri[3, 2] - cotri[1, 2]]

        detJ = abs(det(Jmat))
        JInv = inv(Jmat)

        # Quadrature points and weights
        quad_pts, quad_wghts = quad_168()
        nqpts = size(quad_pts, 1)

        # Transform quadrature points to the actual triangle
        xy_pts = (Jmat * quad_pts')'
        xy_pts[:, 1] .+= cotri[1, 1]
        xy_pts[:, 2] .+= cotri[1, 2]
        quad_wghts .*= detJ

        # Evaluate the scalar function at quadrature points
        xpts = xy_pts[:, 1]'
        ypts = xy_pts[:, 2]'
        scalvals = VV.(xpts, ypts)

        # Velocity and additional scalar contributions
        Vel1 = GlobalV[Vstart] .^ 2
        scalvals2 = bet * (Vel1' * ten1a)
        scalvals .+= scalvals2

        # Compute the local mass matrix
        LMass = inner_prod_ten1_ten1_scal(nodeco, elnode,itrg,scalvals, ten1a, quad_wghts')

        # Assemble into global sparse structure
        for count in 1:size(GlTrgVe, 1)
            for count1 in 1:size(GlTrgVe, 1)
                number2 += 1
                if number2 > length(i2)
                    error("Preallocated arrays are too small; increase their size.")
                end
                i2[number2] = GlTrgVe[count]
                j2[number2] = GlTrgVe[count1]
                m1[number2] = LMass[count, count1]
            end
        end
    end

    # Construct sparse matrix
    MassMatrixGPE = sparse(i2[1:number2], j2[1:number2], m1[1:number2], 2 * NVU, 2 * NVU)
    MassMatrixGPE = MassMatrixGPE[1:2:end, 1:2:end]

    # Restore original GlobalV
    GlobalV = GlobalV2

    return MassMatrixGPE
end


function GPE_Picard_Iteration(GlobalV, elnode, nodeco, basisValues, NVU, 
    nVert, bdrydof, StiffnessMatrix, MassMatrix)
# Assume GPEmats is called elsewhere to define Acoeff, RHSvec, etc.
MassMatrixGPE = computeMassMatrixGPE(elnode, nodeco, GlobalV, basisValues, NVU, nVert)
# Acoeff = 1/2 * StiffnessMatrix + MassMatrixGPE
Acoeff = 0.5 * StiffnessMatrix + MassMatrixGPE

# RHSvec = MassMatrix * GlobalV
RHSvec = MassMatrix * GlobalV

# Dirichlet BC: Apply Dirichlet boundary conditions for velocity
bdrydirvals = zeros(length(bdrydof))  # Assuming `bdrydof` contains the indices for Dirichlet BC
RHSvec .-= Acoeff[:, bdrydof] * bdrydirvals
Acoeff[:, bdrydof] .= 1e-30  # Set the appropriate columns to a small value
Acoeff[bdrydof, :] .= 1e-30  # Set the appropriate rows to a small value

# Set diagonal entries of Acoeff for Dirichlet BC
for ii in 1:size(bdrydof, 1)
    Acoeff[bdrydof[ii], bdrydof[ii]] = 1
end

# Update RHSvec with Dirichlet boundary condition values
RHSvec[bdrydof] .= bdrydirvals

# Solve the linear system for GlobalV
GlobalV .= Acoeff \ RHSvec

# Normalize GlobalV
normPhiHat = sqrt(GlobalV' * (MassMatrix * GlobalV))
mu = 1 / normPhiHat

# Normalize the solution
GlobalV .= GlobalV / normPhiHat
return GlobalV
end

function p4_f!(G,u, elnode, nodeco, basisValues, NVU, 
    nVert, bdrydof, StiffnessMatrix, MassMatrix)
    u2 = copy(u)
    G .= GPE_Picard_Iteration(u2, elnode, nodeco, basisValues, NVU, 
        nVert, bdrydof, StiffnessMatrix, MassMatrix)
end

#32, 128, 256
function P4(meshfile)
    
    meshfile=pkgdir(AndersonPlus,"data","GrossPitvaeskiiEqn",meshfile);
    nodeco, elnode, bdynde, nVert = FFEM2ErvGrd(meshfile);

    nodeco, elnode, bdynde, x_0, NVU, basisValues, gradBasisValues, detJglobal, 
    quad_wghtsglobal, dirbdynde, bdrydof = GPESetup(nodeco, elnode, bdynde,nVert);

    StiffnessMatrix, MassMatrix = createMassStiffness(elnode, nodeco, 
        basisValues, gradBasisValues, NVU, nVert);



    return AAProblem((G,u) -> p4_f!(G,u, elnode, nodeco, basisValues, NVU, 
    nVert, bdrydof, StiffnessMatrix, MassMatrix),
            x_0,
            AAConvParams(1e-10, 0))
end

    
