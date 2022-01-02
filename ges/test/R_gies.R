library(pcalg)
path_out = '/Users/olga/PycharmProjects/gies2/ges/test/~'
data = read.csv('/Users/olga/PycharmProjects/gies2/ges/test/data', header = FALSE)
targets <- list(integer(0), 2, 3)
target.index <- c(rep(1, 10000), rep(2, 10000), rep(3, 10000))

score = new("GaussL0penIntScore", data, targets, target.index, intercept = TRUE)
gies.fit <- gies(score)

A <- matrix(0, ncol(data), ncol(data))
A[1, 3] <- 1
A[2, 3] <- 1
A[3, 4] <- 1
A[3, 5] <- 1
A[4, 5] <- 1

A_0 <- matrix(0, ncol(data), ncol(data))

score_empty_graph <- score$global.score(as(A_0, "GaussParDAG"))
score$global.score(as(A_0, "GaussParDAG"))
score_gies <- score$global.score(gies.fit$repr)
score_true <- score$global.score(as(A, "GaussParDAG"))
score$pp.dat$scatter[[score$pp.dat$scatter.index[1]]]/score$pp.dat$data.count[1]


scores <- c(score_empty_graph, score_gies, score_true)
write.table(scores,'/Users/olga/PycharmProjects/gies2/ges/test/scores.csv',row.names=FALSE,col.names=FALSE)
