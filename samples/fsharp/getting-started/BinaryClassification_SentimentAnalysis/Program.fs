module BinaryClassification_SentimentAnalysis

open System
open System.IO

open Microsoft.ML.Runtime.Api
open Microsoft.ML.Data

open Microsoft.ML.Data
open Microsoft.ML.Transforms
open Microsoft.ML.Trainers
open Microsoft.ML.Runtime.Data
open Microsoft.ML.Runtime.Learners

[<CLIMutable>]
type SentimentData =
    { [<Column("0")>]
      SentimentText: string
      [<Column("1", name="Label")>]
      Sentiment: float32 }

[<CLIMutable>]
type SentimentPrediction =
    { [<ColumnName("PredictedLabel")>]
      Sentiment: bool }


//type SentimentData() =
//    [<Column("0")>]
//    member val SentimentText: string = "" with get, set

//    [<Column("1", name="Label")>]
//    member val Sentiment: float32 = 0.0f with get, set

//type SentimentPrediction() =
//    [<ColumnName("PredictedLabel")>]
//    member val Sentiment: bool = false with get, set


let sentiments = 
   [| { SentimentText = "Contoso's 11 is a wonderful experience"; Sentiment = 1.0f }
      { SentimentText = "The acting in this movie is very bad"; Sentiment = 0.0f }
      { SentimentText = "Joe versus the Volcano Coffee Company is a great film."; Sentiment = 1.0f } |]

let AppPath = Path.Combine(__SOURCE_DIRECTORY__, "../../../..")
let TrainDataPath = Path.Combine(AppPath, "datasets", "sentiment-imdb-train.txt")
let TestDataPath = Path.Combine(AppPath, "datasets", "sentiment-yelp-test.txt")
let modelPath = Path.Combine(AppPath, "SentimentModel.zip")

let TrainAsync() =
    // LocalEnvironment holds all steps of the learning process: data, transforms, learners.  
    use env = new LocalEnvironment()

    // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
    // all the column names and their types.
    //pipeline.Add(TextLoader(TrainDataPath).CreateFrom<SentimentData>())
    let col1 = new TextLoader.Column("Label", Nullable DataKind.Bool  , 0)
    let col2 = new TextLoader.Column("Text", Nullable DataKind.Text  , 1)
    //col1.
    
    let textLoaderArgs = new TextLoader.Arguments()
    textLoaderArgs.Separator <- "tab"
    textLoaderArgs.HasHeader <- true
    textLoaderArgs.Column <- [|col1 ; col2|] 

    let reader = new TextLoader(env, textLoaderArgs)


    //Load training data
    let trainingDataView = reader.Read(new MultiFileSource(TrainDataPath))

    let pipeline = new TextTransform(env, "Text", "Features")
        pipeline.Append
        .env, new LinearClassificationTrainer.Arguments(), "Text", "Features")
    // TextFeaturizer is a transform that will be used to featurize an input column to format and clean the data.
    pipeline.Add(TextFeaturizer("Features", "SentimentText"))

    // FastTreeBinaryClassifier is an algorithm that will be used to train the model.
    // It has three hyperparameters for tuning decision tree performance. 
    pipeline.Add(FastTreeBinaryClassifier(NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2))

    printfn "=============== Training model ==============="
    // The pipeline is trained on the dataset that has been loaded and transformed.
    let model = pipeline.Train<SentimentData, SentimentPrediction>()

    // Saving the model as a .zip file.
    model.WriteAsync(modelPath) |> Async.AwaitTask |> Async.RunSynchronously

    printfn "=============== End training ==============="
    printfn "The model is saved to %s" modelPath

    model

let Evaluate(model: PredictionModel<SentimentData, SentimentPrediction> ) =
    // To evaluate how good the model predicts values, the model is ran against new set
    // of data (test data) that was not involved in training.
    let testData = TextLoader(TestDataPath).CreateFrom<SentimentData>()
    
    // BinaryClassificationEvaluator performs evaluation for Binary Classification type of ML problems.
    let evaluator = BinaryClassificationEvaluator()

    printfn "=============== Evaluating model ==============="

    let metrics = evaluator.Evaluate(model, testData)
    // BinaryClassificationMetrics contains the overall metrics computed by binary classification evaluators
    // The Accuracy metric gets the accuracy of a classifier which is the proportion 
    //of correct predictions in the test set.

    // The Auc metric gets the area under the ROC curve.
    // The area under the ROC curve is equal to the probability that the classifier ranks
    // a randomly chosen positive instance higher than a randomly chosen negative one
    // (assuming 'positive' ranks higher than 'negative').

    // The F1Score metric gets the classifier's F1 score.
    // The F1 score is the harmonic mean of precision and recall:
    //  2 * precision * recall / (precision + recall).

    printfn "Accuracy: %0.2f" metrics.Accuracy
    printfn "Auc: %0.2f" metrics.Auc
    printfn "F1Score: %0.2f" metrics.F1Score
    printfn "=============== End evaluating ==============="
    printfn ""

// STEP 1: Create a model
let model = TrainAsync()

// STEP2: Test accuracy
Evaluate(model)

// STEP 3: Make a prediction
let predictions = model.Predict(sentiments)

//for (sentiment, prediction) in Seq.zip sentiments predictions do
    //printfn "Sentiment: %s | Prediction: %s sentiment" sentiment.SentimentText (if prediction.Sentiment then "Positive" else "Negative")

//TODO: which way is better? above or below?
predictions
|> Seq.zip sentiments
|> Seq.iter (fun (s,p) -> printfn "Sentiment: %s | Prediction: %s sentiment" s.SentimentText (if p.Sentiment then "Positive" else "Negative"))

Console.ReadLine() |> ignore

