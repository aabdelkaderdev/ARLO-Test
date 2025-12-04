// See https://aka.ms/new-console-template for more information
using Research.DiscArch.Console;
using Research.DiscArch.Designer;
using Research.DiscArch.Models;
using Research.DiscArch.TestData;

//var experiment = new OprimizationExperiment();
//experiment.Run();

//var experiment = new CategorizedReqsExperiment();
//experiment.Run(Research.DiscArch.TestData.SystemNames.Messaging);

// Interactive requirements file selection
var selectedFile = ResourceManager.SelectRequirementsFile();
Console.WriteLine();

var experimentSettings = new ExperimentSettings
{
    System = SystemNames.SpringXd, // Default system, but requirements will be loaded from selected file
    OptimizationStrategy = Enum.GetName(OptimizerMode.ILP),
    QualityWeightsMode = Enum.GetName(QualityWeightsMode.Inferred),
    LoadReqsFromFile = true,
    LoadConditionGroupsFromFile = false,
    MatrixPath = "quality_archipattern_matrix_bal.csv",
    SelectedRequirementsFile = selectedFile // New property for the selected file
};

//var experiment = new CategorizedReqsExperiment(experimentSettings);
//var experiment = new VaryingRequirementsExpermeriment(experimentSettings);

//var experiment = new ProvidedQAWeightsExperiment();
//experiment.Run(Research.DiscArch.TestData.SystemNames.OfficerDispatcher);

//Targetting SQL for DB
experimentSettings.RemovalStratgy.Kind = RemovalStrategy.RemovalKind.Targeted;
experimentSettings.RemovalStratgy.DesiredQAs.Add("Security");
experimentSettings.RemovalStratgy.DesiredQAs.Add("Maintainability");
experimentSettings.RemovalStratgy.DesiredQAs.Add("Cost Efficiency");
experimentSettings.RemovalStratgy.UndesiredQAs.Add("Performance Efficiency");
experimentSettings.RemovalStratgy.UndesiredQAs.Add("Reliability");

//var experiment = new CategorizedReqsExperiment(experimentSettings);
//var experiment = new VaryingRequirementsExpermeriment(experimentSettings);
//experiment.Run();

var experiment = new IdentifyingInflunetialSetsExperiment(experimentSettings);
experiment.Run();

Console.ReadLine();

