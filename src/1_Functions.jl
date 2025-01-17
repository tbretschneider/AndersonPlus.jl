function greet_your_package_name()
    return "Hello YourPackageName!"
end

function AASolve(input::AAInput)::AAOutput

    x0 = input.problem.x0
    GFix! = input.problem.GFix!
    convparams = input.problem.convparams

    method = input.algorithm.method
    algorithmparams = input.algorithm.algorithmparams

    liveanalysis = input.analyses.liveanalysis
    midanalysis = input.analyses.midanalysis
    postanalysis = input.analyses.postanalysis
    updatefreq = input.analyses.updatefreq

    midanalysisfunc = create_midanalysis_function(midanalysis)
    liveanalysisfunc = create_liveanalysis_function(liveanalysis)

    nextiterate! = create_next_iterate_function(GFix!, method, liveanalysisfunc, midanalysisfunc)

    iterations = 1
    
    converged = false


    startwalltime = time()

    output_liveanalysis(liveanalysis, iterations, updatefreq, startwalltime)

    x_k = copy(x_0)
    x_kp1 = copy(x_0)

    postanalysis = AAAnalysisOutput()
    return AAOutput(solution,input,postanalysis)
end