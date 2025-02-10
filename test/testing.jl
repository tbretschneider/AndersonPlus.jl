using AndersonPlus

# Parameters
k0 = 8.0
N = 1000
ε = 0.2

Problem = P3(k0, ε, N)

Algorithm = AAAlgorithm(AAMethod(:quickaa,(m=10, threshold_func = (positions, iterations) -> fill(1.0, length(positions)))),
                        (maxit = 20, ))

Analyses = AAAnalysis([],
                    [:HisStuf,:alpha],
                    0,false)
                
Output = AASolve(AAInput(Problem,Algorithm,Analyses))