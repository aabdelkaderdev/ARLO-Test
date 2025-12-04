# ARLO Project

## Overview
ARLO is a C# project that converts software requirements into architectural decisions. Utilizing DeepSeek's deepseek-chat model for parsing and an ILP-based optimizer for decision-making streamlines the architectural design process.

## Key Components
- **RequirementParser**: Parses natural language requirements into ASRs.
- **Architect**: Develops architectural policies from ASRs.

## How to Use
1. **Clone the Repository**: Access the code on GitHub.
2. **Register your DeepSeek API Key**: See below.
3. **Install Dependencies**: Follow the setup instructions for necessary libraries and tools.
4. **Run the Console Application**: The program will display available requirements files for selection.
5. **Select a Requirements File**: Enter the number corresponding to your chosen requirements file.
6. **Review Output**: Analyze the generated architectural decisions.

## Interactive Requirements File Selection

When you run the ARLO console application, you will be presented with a numbered list of available requirements files:

```
Available Requirements Files:
=============================
1. MessagingReqs.txt
2. KaggleReq.txt
3. ContradictoryReqs.txt
4. AppFlowy.txt
5. MLonBL.txt
6. OfficerDispatcher.txt
7. Bamboo.txt
8. Mule.txt
9. aptana.txt
10. springxd.txt
11. SmallMessagingSystem.txt

Enter the number of the requirements file to use: 
```

Simply enter the number corresponding to the requirements file you want to analyze, and the program will process that file through the experiment.

## Contributing
Contributions are welcome! Please take a look at our contributing guidelines for more information.

### Setting Up Your DeepSeek API Key

ARLO uses the DeepSeek API for AI-powered requirement analysis. To set up your API key:

#### On macOS/Linux

1. **Open Terminal**: Open the Terminal application.

2. **Edit Shell Configuration File**:
    - Identify the shell you are using by running:
      ```sh
      echo $SHELL
      ```
    - If you are using `bash`, open:
      ```sh
      nano ~/.bash_profile
      ```
      or
      ```sh
      nano ~/.bashrc
      ```
    - If you are using `zsh` (default in macOS Catalina and later), open:
      ```sh
      nano ~/.zshrc
      ```

3. **Set the Environment Variable**:
    - Add the following line to the configuration file you opened:
      ```sh
      export DeepSeekApiKey="your_deepseek_api_key_here"
      ```
    - Replace `"your_deepseek_api_key_here"` with your actual DeepSeek API key.

4. **Save and Close the File**:
    - If you are using `nano`, save the file by pressing `CTRL + O`, then press `Enter` to confirm. Exit by pressing `CTRL + X`.

5. **Apply the Changes**:
    - Run the following command in the Terminal to apply the changes:
      ```sh
      source ~/.bash_profile
      ```
      or
      ```sh
      source ~/.bashrc
      ```
      or
      ```sh
      source ~/.zshrc
      ```

6. **Verify the Environment Variable**:
    - Verify that the environment variable is set correctly by running:
      ```sh
      echo $DeepSeekApiKey
      ```
    - This should print your DeepSeek API key to the terminal.

#### On Windows

1. **Open System Properties**: Right-click on "This PC" or "Computer" and select "Properties", then click "Advanced system settings".

2. **Environment Variables**: Click the "Environment Variables" button.

3. **Add New Variable**: Under "User variables" or "System variables", click "New" and add:
    - Variable name: `DeepSeekApiKey`
    - Variable value: `your_deepseek_api_key_here`

4. **Apply Changes**: Click "OK" to save and close all dialogs.

5. **Restart your IDE/Terminal**: You may need to restart your development environment for the changes to take effect.

Now, when you run your C# application, the environment variable `DeepSeekApiKey` will be available, and the application will retrieve it using:

```csharp
string apiKey = Environment.GetEnvironmentVariable("DeepSeekApiKey");
```

## API Information

ARLO uses the DeepSeek API with the following configuration:
- **API Endpoint**: `https://api.deepseek.com/chat/completions`
- **Model**: `deepseek-chat`

**Note**: The embeddings functionality is not currently supported as DeepSeek does not provide an embeddings API. If you need embeddings functionality, consider using an alternative embeddings provider.
