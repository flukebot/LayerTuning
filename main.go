package main

import (
	"dense"
	"fmt"
	"math"
	"math/rand"
	"time"
)

const targetText = "hi my name is sam please"

var tokenToIndex map[int]int
var indexToToken map[int]int

func init() {
	// Build custom vocabulary using GPT-3 tokenizer
	text := "1 " + targetText // Include the input "1" and the target text
	tokens := dense.TokenizeInput(text)
	tokenSet := make(map[int]struct{})
	for _, token := range tokens {
		tokenSet[token] = struct{}{}
	}
	// Map GPT-3 tokens to custom indices
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
	// Adjust output size to accommodate vocab size per position
	totalOutputSize := outputSize * vocabSize

	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Define initial output activation types for each output neuron
	outputActivationTypes := dense.GenerateActivationTypes(totalOutputSize, "linear") // Use linear activation; softmax will be applied in loss

	// Initialize the network configuration with updated parameters
	modelConfig := dense.CreateCustomNetworkConfig(inputSize, hiddenNeurons, totalOutputSize, outputActivationTypes, "Model_Initial", projectName)

	// Training parameters
	numEpochs := 5000
	learningRate := 0.01
	mutationRate := 5 // Mutation rate percentage for mutations

	// Variables for monitoring training progress
	bestLoss := math.Inf(1)
	noImprovementEpochs := 0
	patience := 200 // Number of epochs to wait before adding mutations

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

		// Check for improvement
		if loss < bestLoss {
			bestLoss = loss
			noImprovementEpochs = 0
		} else {
			noImprovementEpochs++
		}

		// Backward pass and weight updates
		dense.Backpropagate(modelConfig, activations, lossGradients, learningRate)

		// Apply mutations every 'patience' epochs without improvement
		if noImprovementEpochs >= patience {
			fmt.Printf("No improvement in loss for %d epochs. Applying mutations.\n", patience)
			dense.MutateNetwork(modelConfig, learningRate, mutationRate)
			noImprovementEpochs = 0
			// Optionally, reset bestLoss to current loss
			bestLoss = loss
		}

		// Print progress every 100 epochs
		if epoch%100 == 0 || epoch == numEpochs {
			generatedOutput := dense.GenerateOutputFromProbabilities(outputValues, vocabSize, indexToToken)
			// Since `generatedOutput` is already a string, no need to decode
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
