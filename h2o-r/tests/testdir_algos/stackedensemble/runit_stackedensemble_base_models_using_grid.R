setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# PUBDEV-4534
stackedensemble_accepts_grid_and_models_as_base_models.test <- function() {
    # This test asserts that base models can be one of these:
    # * list of models
    # * GridSearch
    # * list of GridSearches
    # * list of Gridsearches and models

    get_base_models <- function(ensemble) {
        unlist(lapply(ensemble@parameters$base_models, function (base_model) base_model$name))
    }

    train <- h2o.uploadFile(locate("smalldata/testng/higgs_train_5k.csv"),
                            destination_frame = "higgs_train_5k")

    valid <- h2o.uploadFile(locate("smalldata/testng/higgs_test_5k.csv"),
                            destination_frame = "higgs_test_5k")

    y <- "response"
    x <- setdiff(names(train), y)
    train[,y] <- as.factor(train[,y])
    nfolds <- 5
    my_gbm <- h2o.gbm(x = x,
                      y = y,
                      training_frame = train,
                      distribution = "bernoulli",
                      ntrees = 10,
                      nfolds = nfolds,
                      fold_assignment = "Modulo",
                      keep_cross_validation_predictions = TRUE,
                      seed = 1)

    my_rf <- h2o.randomForest(x = x,
                              y = y,
                              training_frame = train,
                              ntrees = 10,
                              nfolds = nfolds,
                              fold_assignment = "Modulo",
                              keep_cross_validation_predictions = TRUE,
                              seed = 1)


    hyper_params <- list(ntrees = c(3, 5))

    grid1 <- h2o.grid("gbm", x = x, y = y,
                      training_frame = train,
                      validation_frame = valid,
                      seed = 1,
                      nfolds = nfolds,
                      fold_assignment = "Modulo",
                      keep_cross_validation_predictions = TRUE,
                      hyper_params = hyper_params)

    grid2 <- h2o.grid("drf", x = x, y = y,
                      training_frame = train,
                      validation_frame = valid,
                      seed = 1,
                      nfolds = nfolds,
                      fold_assignment = "Modulo",
                      keep_cross_validation_predictions = TRUE,
                      hyper_params = hyper_params)

    # List of model
    stack0 <- h2o.stackedEnsemble(x = x,
                                  y = y,
                                  training_frame = train,
                                  base_models = list(my_gbm))
    expect_equal(get_base_models(stack0), my_gbm@model_id)

    # Single Grid
    stack1 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = grid1
    )

    stack2 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = grid1@model_ids
    )

    expect_equal(get_base_models(stack1), get_base_models(stack2))

    # List of Grids
    stack3 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = list(grid1, grid2)
    )

    stack4 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = as.list(c(grid1@model_ids, grid2@model_ids))
    )

    expect_equal(get_base_models(stack3), get_base_models(stack4))

    # List of grids and models
    stack5 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = list(grid1, my_gbm, grid2)
    )

    stack6 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = as.list(c(grid1@model_ids, my_gbm@model_id, grid2@model_ids))
    )

    expect_equal(get_base_models(stack5), get_base_models(stack6))

    # List of grids and models using model_id
    stack7 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = list(grid1, my_gbm@model_id, grid2)
    )

    stack8 <- h2o.stackedEnsemble(x=x,
                                  y=y,
                                  training_frame = train,
                                  base_models = as.list(c(grid1@model_ids, my_gbm@model_id, grid2@model_ids))
    )

    expect_equal(get_base_models(stack7), get_base_models(stack8))
}

doTest("Stacked Ensemble accept both models and grid as base models", stackedensemble_accepts_grid_and_models_as_base_models.test)
