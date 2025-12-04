using System.Text;
using System.Text.Json;
using Newtonsoft.Json;

namespace Research.DiscArch.Services;

public class DeepSeekEmbedding
{
    [JsonProperty("model")]
    public string Model { get; set; }

    [JsonProperty("input")]
    public string[] Input { get; set; }
}

public class ChatMessage
{
    [JsonProperty("role")]
    public string Role { get; set; }

    [JsonProperty("content")]
    public string Content { get; set; }
}

public class Choice
{
    [JsonProperty("message")]
    public ChatMessage Data { get; set; }
}

public class DeepSeekEmbeddingResponse
{
    [JsonProperty("data")]
    public List<EmbeddingData> Data { get; set; }
}

public class EmbeddingData
{
    [JsonProperty("embedding")]
    public List<double> Embedding { get; set; }
}

public class GptService
{
    private readonly HttpClient client;
    private readonly string apiKey;
    public const int MaxWordPerCall = 5000; //8000 tokens and 0.7 word per token 

    public GptService()
    {
        client = new HttpClient();
        apiKey = Environment.GetEnvironmentVariable("DeepSeekApiKey");
    }

    public async Task<string> Call(string instruction, string ask)
    {
        int delay = 1000;
        int retryCount = 0;
        int maxRetries = 5;
        string apiUrl = "https://api.deepseek.com/chat/completions";
        client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);

        var requestBody = new
        {
            model = "deepseek-chat",
            messages = new[]
            {
                new { role = "system", content = instruction },
                new { role = "user", content = ask }
            }
        };

        string jsonRequestBody = System.Text.Json.JsonSerializer.Serialize(requestBody);
        var content = new StringContent(jsonRequestBody, Encoding.UTF8, "application/json");

        while (retryCount < maxRetries)
        {
            var response = await client.PostAsync(apiUrl, content);

            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(responseContent);
                return doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString();
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
            {
                // Wait and retry
                await Task.Delay(delay);
                delay *= 2; // Exponential backoff
                retryCount++;
            }
            else
            {
                Console.WriteLine("DeepSeek Error: " + response.ReasonPhrase);
                throw new Exception("Error calling DeepSeek Chat API: " + response.ReasonPhrase);
            }
        }

        throw new Exception("Error calling DeepSeek Chat API: Too many requests");
    }

    public async Task<List<List<double>>> GetEmbeddings(List<string> conditions)
    {
        // Note: DeepSeek does not currently provide an embeddings API.
        // This method is preserved for compatibility but will throw an exception if called.
        // Consider using an alternative embeddings provider if this functionality is needed.
        throw new NotSupportedException("DeepSeek does not currently provide an embeddings API. Consider using an alternative embeddings provider.");
    }
}
