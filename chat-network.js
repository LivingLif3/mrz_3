const { create, all } = require("mathjs");
const math = create(all);

class ElmanNetwork {
    constructor(inputNeurons, hiddenNeurons, outputNeurons, learningRate) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.learningRate = learningRate;

        // Инициализация весов
        this.inputWeights = math.random([inputNeurons, hiddenNeurons], -1, 1);
        this.hiddenWeights = math.random([hiddenNeurons, hiddenNeurons], -1, 1);
        this.outputWeights = math.random([hiddenNeurons, outputNeurons], -1, 1);

        // Начальное скрытое состояние
        this.hiddenState = math.zeros(hiddenNeurons);
    }

    activate(x) {
        return math.tanh(x); // Гиперболический тангенс как функция активации
    }

    activateDerivative(x) {
        return 1 - math.square(math.tanh(x)); // Производная гиперболического тангенса
    }

    forward(input) {
        // Вычисление нового скрытого состояния
        const inputHidden = math.multiply(input, this.inputWeights);
        const hiddenHidden = math.multiply(this.hiddenState, this.hiddenWeights);
        this.hiddenState = this.activate(math.add(inputHidden, hiddenHidden));

        // Вычисление выхода
        const output = math.multiply(this.hiddenState, this.outputWeights);
        return output;
    }

    backward(input, target, output) {
        const outputError = math.subtract(target, output); // Ошибка на выходе
        const outputGradient = math.dotMultiply(outputError, this.activateDerivative(output));

        // Обновление весов между скрытым слоем и выходом
        const hiddenGradient = math.multiply(outputGradient, math.transpose(this.outputWeights));
        this.outputWeights = math.add(
            this.outputWeights,
            math.multiply(math.transpose(this.hiddenState), math.multiply(outputGradient, this.learningRate))
        );

        // Обновление весов между скрытым слоем и входами
        const hiddenError = math.dotMultiply(hiddenGradient, this.activateDerivative(this.hiddenState));
        this.inputWeights = math.add(
            this.inputWeights,
            math.multiply(math.transpose(input), math.multiply(hiddenError, this.learningRate))
        );

        // Обновление весов скрытого состояния
        this.hiddenWeights = math.add(
            this.hiddenWeights,
            math.multiply(math.transpose(this.hiddenState), math.multiply(hiddenError, this.learningRate))
        );
    }

    train(inputs, targets, epochs) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                this.backward(inputs[i], targets[i], output);
                totalError += math.sum(math.square(math.subtract(targets[i], output)));
            }
            console.log(`Epoch ${epoch + 1}, Error: ${totalError}`);
        }
    }

    predict(input) {
        return this.forward(input);
    }
}

class SequenceGenerator {
    generateFibonacciByIndices(startIndex, endIndex) {
        if (startIndex < 0 || endIndex < 0) {
            throw new Error("Indices must be non-negative.");
        }
        if (startIndex > endIndex) {
            throw new Error("Start index cannot be greater than end index.");
        }

        const fibonacciNumbers = [];
        let a = 0, b = 1;

        for (let i = 0; i <= endIndex; i++) {
            if (i >= startIndex) {
                fibonacciNumbers.push(a);
            }
            const next = a + b;
            a = b;
            b = next;
        }

        return fibonacciNumbers;
    }
}

// Генерация данных
const sequenceGenerator = new SequenceGenerator();
const sequence = sequenceGenerator.generateFibonacciByIndices(0, 10);

// Формирование данных методом скользящего окна
const windowSize = 3;
const inputs = [];
const targets = [];
for (let i = 0; i < sequence.length - windowSize; i++) {
    inputs.push(sequence.slice(i, i + windowSize));
    targets.push([sequence[i + windowSize]]);
}

// Преобразование данных в формат, подходящий для math.js
const normalizedInputs = inputs.map(input => input.map(num => num / 100));
const normalizedTargets = targets.map(target => target.map(num => num / 100));

// Создание и обучение сети
const network = new ElmanNetwork(windowSize, 5, 1, 0.01);
network.train(normalizedInputs, normalizedTargets, 1000);

// Предсказание следующего числа
const testInput = [sequence.slice(-windowSize)].map(input => input.map(num => num / 100));
const prediction = network.predict(testInput[0]);

console.log("Original sequence:", sequence);
console.log("Prediction:", prediction[0] * 100);
