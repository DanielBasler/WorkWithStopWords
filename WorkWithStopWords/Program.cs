using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace WorkWithStopWords
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var emptyData = new List<TextData>();
            var data = context.Data.LoadFromEnumerable(emptyData);

            //var pipeline = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', '.', ',' })
            //   .Append(context.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens",
            //       Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.German));

            var pipeline = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', '.', ',' })
                .Append(context.Transforms.Text.RemoveStopWords("Tokens", "Tokens", new[] { "bis", "die" }));

            var stopWords = pipeline.Fit(data);

            var engine = context.Model.CreatePredictionEngine<TextData, TextTokens>(stopWords);

            var newText = engine.Predict(new TextData { Text = "Es dauert ziemlich lange bis die Seite gedruckt wird." });

            var sb = new StringBuilder();

            foreach (var token in newText.Tokens)
            {
                sb.AppendLine(token);
            }

            Console.WriteLine(sb.ToString());
            Console.ReadLine();
        }        
    }

    public class TextTokens
    {
        public string[] Tokens { get; set; }
    }

    public class TextData
    {
        public string Text { get; set; }
    }
}
