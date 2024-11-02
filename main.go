package main

import (
	"dense"
	"fmt"
)

func main() {
	fmt.Println("Setting up text generation model with dynamic softmax output layer.")

	// Define model parameters
	projectName := "AIModelTextGeneration"
	inputSize := 1              // Single token input for demonstration
	numFirstLayerNeurons := 128 // Number of neurons in the first hidden layer
	outputSize := 1000          // Output size, targeting 1000 tokens for text generation

	// Generate softmax activation types for text generation in the output layer
	outputTypes := dense.GenerateActivationTypes(outputSize)

	// Create the network configuration
	config := dense.CreateCustomNetworkConfig(inputSize, numFirstLayerNeurons, outputSize, outputTypes, "Model_123", projectName)

	// Example input tokenization
	encodedInput := dense.TokenizeInput("1")
	fmt.Println("Encoded Input:", encodedInput)

	// Feedforward pass through the network using the encoded input
	inputMap := map[string]interface{}{"input0": float64(encodedInput[0])}
	output := dense.Feedforward(config, inputMap)

	// Decode the model's output back into text
	outputTokens := make([]float64, len(output))
	for _, v := range output {
		outputTokens = append(outputTokens, v)
	}
	outputText := dense.DecodeTokens(outputTokens)
	fmt.Println("Generated Output:", outputText)
}
