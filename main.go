package main

import (
	"dense"
	"fmt"
	"math/rand"
	"time"
)

const targetText = "hi my name is sam please"

var tokenToIndex map[int]int
var indexToToken map[int]int

func init() {
	// Build custom vocabulary
	text := "1 " + targetText // Include the input "1" and the target text
	tokens := dense.TokenizeInput(text)
	tokenSet := make(map[int]struct{})
	for _, token := range tokens {
		tokenSet[token] = struct{}{}
	}
	// Map tokens to indices
	tokenToIndex = make(map[int]int)
	indexToToken = make(map[int]int)
	i := 0
	for token := range tokenSet {
		tokenToIndex[token] = i
		indexToToken[i] = token
		i++
	}
}

func main() {
	fmt.Println("Starting text generation model...")

	// Initialize model parameters
	projectName := "TextGenerationProject"
	inputSize := 1
	hiddenNeurons := 128

	// Tokenize target text
	targetTokens := dense.TokenizeInputCustom(targetText, tokenToIndex)
	outputSize := len(targetTokens)
	vocabSize := len(tokenToIndex)

	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Define initial output activation types for each output neuron
	outputActivationTypes := dense.GenerateActivationTypes(outputSize*vocabSize, "softmax") // Each position has vocabSize outputs

	// Initialize the network configuration
	modelConfig := dense.CreateCustomNetworkConfig(inputSize, hiddenNeurons, outputSize*vocabSize, outputActivationTypes, "Model_Initial", projectName)

	// Training parameters
	numEpochs := 1000
	learningRate := 0.01

	// Tokenize input (e.g., "1")
	inputToken := dense.TokenizeInputCustom("1", tokenToIndex)
	if len(inputToken) == 0 {
		fmt.Println("Error: Input contains no valid tokens.")
		return
	}

	// Training loop
	for epoch := 1; epoch <= numEpochs; epoch++ {
		// Forward pass
		inputMap := map[string]float64{"input0": float64(inputToken[0])}
		outputValues, activations := dense.FeedforwardWithActivations(modelConfig, inputMap)

		// Compute loss and accuracy
		loss, lossGradients := dense.ComputeCrossEntropyLossAndGradients(outputValues, targetTokens, vocabSize)
		accuracy := dense.ComputeTokenAccuracy(outputValues, targetTokens, vocabSize)

		// Backward pass and weight updates
		dense.Backpropagate(modelConfig, activations, lossGradients, learningRate)

		// Print progress every 100 epochs
		if epoch%100 == 0 || epoch == numEpochs {
			generatedOutput := dense.GenerateOutputFromProbabilities(outputValues, vocabSize, indexToToken)
			fmt.Printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%% | Output: %s\n", epoch, loss, accuracy*100, generatedOutput)
		}
	}

	// Test the model with the input "1"
	fmt.Println("\nTesting the trained model with input '1'...")
	inputMap := map[string]float64{"input0": float64(inputToken[0])}
	outputValues, _ := dense.FeedforwardWithActivations(modelConfig, inputMap)
	generatedOutput := dense.GenerateOutputFromProbabilities(outputValues, vocabSize, indexToToken)
	fmt.Printf("Generated Output: %s\n", generatedOutput)
}
