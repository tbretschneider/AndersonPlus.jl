function greet_your_package_name()
    return "Hello YourPackageName!"
end

function AASolve(input::AAInput)::AAOutput

    x_0 = input.problem.x0
    GFix! = input.problem.GFix!
    convparams = input.problem.convparams

    method = input.algorithm.method
    algorithmparams = input.algorithm.algorithmparams

    maxit = algorithmparams.maxit

    liveanalysis = input.analyses.liveanalysis
    midanalysis = input.analyses.midanalysis
    updatefreq = input.analyses.updatefreq
    summary = input.analyses.summary

    midanalysisfunc = create_midanalysis_function(midanalysis)
    liveanalysisfunc = create_liveanalysis_function(liveanalysis)

    calcx_kp1! = create_next_iterate_function(GFix!, method, liveanalysisfunc, midanalysisfunc)

    converged = false

    startwalltime = time()

    x_k = copy(x_0)
    x_kp1 = copy(x_0)

    HS = initialise_historicalstuff(method.methodname,x_k)

    fullmidanalysis = []

    while (HS.iterations < maxit) && ~converged
        midanalysis, liveanalysis = calcx_kp1!(HS,x_kp1,x_k)

        push!(fullmidanalysis,midanalysis)

        output_liveanalysis(liveanalysis, HS.iterations, updatefreq, startwalltime)

        converged = checktolerances(x_kp1,x_k,convparams)

        x_k, x_kp1 = x_kp1, x_k
    end

    postanalysis = AAAnalysisOutput(input,fullmidanalysis,HS.iterations)

    output_postanalysis(postanalysis.output,summary)

    solution = x_k

    return AAOutput(solution,input,postanalysis)
end
