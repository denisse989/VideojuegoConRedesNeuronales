"use strict";

const LOG_ON = true; 
const LOG_FREQ = 20000; // Iteraciones de muestra de error

class NeuralNetwork {
  constructor(numInputs, numHidden, numOutputs) {
    this._inputs = [];
    this._hidden = [];
    this._numInputs = numInputs;
    this._numHidden = numHidden;
    this._numOutputs = numOutputs;
    this._bias0 = new Matrix(1, this._numHidden);
    this._bias1 = new Matrix(1, this._numOutputs);
    this._weights0 = new Matrix(this._numInputs, this._numHidden);
    this._weights1 = new Matrix(this._numHidden, this._numOutputs);

    this._outputs=[];

    // Muestra de error
    this._logCount = LOG_FREQ;

    // Randomizar los pesos y bias
    this._bias0.randomWeights();
    this._bias1.randomWeights();
    this._weights0.randomWeights();
    this._weights1.randomWeights();
  }

  get inputs() {
    return this._inputs;
  }

  set inputs(inputs) {
    this._inputs = inputs;
  }

  get hidden() {
    return this._hidden;
  }

  set hidden(hidden) {
    this._hidden = hidden;
  }

  get bias0() {
    return this._bias0;
  }

  set bias0(bias) {
    this._bias0 = bias;
  }

  get bias1() {
    return this._bias1;
  }

  set bias1(bias) {
    this._bias1 = bias;
  }

  get weights0() {
    return this._weights0;
  }

  set weights0(weights) {
    this._weights0 = weights;
  }

  get weights1() {
    return this._weights1;
  }

  set weights1(weights) {
    this._weights1 = weights;
  }

  get logCount() {
    return this._logCount;
  }

  set logCount(count) {
    this._logCount = count;
  }

  getInfo(){

    return [this._bias0, this._bias1, this._hidden, this._inputs, this._outputs, this.weights0, this.weights1];
  }

  feedForward(inputArray) {
    // Convertir el array de entrada en una matriz
    this.inputs = Matrix.convertFromArray(inputArray); 

    // Encontrar los valores en las capas ocultas y aplicar la función de activación
    this.hidden = Matrix.dot(this.inputs, this.weights0);
    this.hidden = Matrix.add(this.hidden, this.bias0); // Aplicar el bias
    this.hidden = Matrix.map(this.hidden, (x) => sigmoid(x));

    // Encontrar los valores de salida y aplicar las funciones de activación
    let outputs = Matrix.dot(this.hidden, this.weights1);
    outputs = Matrix.add(outputs, this.bias1); // Aplicar el bias
    outputs = Matrix.map(outputs, (x) => sigmoid(x));

    this._outputs=outputs;
    return outputs;
  }

  train(inputArray, targetArray) {
    // Obtener los outputs con los inputs
    let outputs = this.feedForward(inputArray);

    //Calcular los errores de entrenamiento
    let targets = Matrix.convertFromArray(targetArray);
    let outputErrors = Matrix.subtract(targets, outputs);

    // Logueo de errores en consolas
    if (LOG_ON) {
      if (this.logCount == LOG_FREQ) {
        console.log("error = " + outputErrors.data[0][0]);
      }
      this.logCount--;
      if (this.logCount == 0) {
        this.logCount = LOG_FREQ;
      }
    }

    // Calculo de deltas (error * derivada de salidas)
    let outputDerivs = Matrix.map(outputs, (x) => sigmoid(x, true));
    let outputDeltas = Matrix.multiply(outputErrors, outputDerivs);

    // Calculo de errores de capas ocultas (producto punto de los deltas por la transpuesta de los pesos))
    let weights1T = Matrix.transpose(this.weights1);
    let hiddenErrors = Matrix.dot(outputDeltas, weights1T);

    // Calculo de deltas ocultos (error * derivada de salidas ocultas)
    let hiddenDerivs = Matrix.map(this.hidden, (x) => sigmoid(x, true));
    let hiddenDeltas = Matrix.multiply(hiddenErrors, hiddenDerivs);

    // Actualizar los pesos (Sumar la transpuesta de los nodos ocultos con producto punto de los deltas ocultos)
    let hiddenT = Matrix.transpose(this.hidden);
    this.weights1 = Matrix.add(
      this.weights1,
      Matrix.dot(hiddenT, outputDeltas)
    );
    let inputsT = Matrix.transpose(this.inputs);
    this.weights0 = Matrix.add(
      this.weights0,
      Matrix.dot(inputsT, hiddenDeltas)
    );

    // actualizar los bias
    this.bias1 = Matrix.add(this.bias1, outputDeltas);
    this.bias0 = Matrix.add(this.bias0, hiddenDeltas);
  }
}

function sigmoid(x, deriv = false) {
  if (deriv) {
    return x * (1 - x); // x = sigmoid(x)
  }
  return 1 / (1 + Math.exp(-x));
}

/***********************
    FUNCIONES MATRICIALES
***********************/

class Matrix {
  constructor(rows, cols, data = []) {
    this._rows = rows;
    this._cols = cols;
    this._data = data;

    // INICIALIZAR la información
    if (data == null || data.length == 0) {
      this._data = [];
      for (let i = 0; i < this._rows; i++) {
        this._data[i] = [];
        for (let j = 0; j < this._cols; j++) {
          this._data[i][j] = 0;
        }
      }
    } else {
      // Revisar los datos y su arquitectura
      if (data.length != rows || data[0].length != cols) {
        throw new Error("Incorrect data dimensions!");
      }
    }
  }

  get rows() {
    return this._rows;
  }

  get cols() {
    return this._cols;
  }

  get data() {
    return this._data;
  }

  // Sumar dos matrices
  static add(m0, m1) {
    Matrix.checkDimensions(m0, m1);
    let m = new Matrix(m0.rows, m0.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        m.data[i][j] = m0.data[i][j] + m1.data[i][j];
      }
    }
    return m;
  }

  // Verificar si ambas matrices tienen las mismas dimensiones
  static checkDimensions(m0, m1) {
    if (m0.rows != m1.rows || m0.cols != m1.cols) {
      throw new Error("Matrices are of different dimensions!");
    }
  }

  // Convertir un array a una matriz 1xN
  static convertFromArray(arr) {
    return new Matrix(1, arr.length, [arr]);
  }

  // Producto punto entre dos matrices
  static dot(m0, m1) {
    if (m0.cols != m1.rows) {
      throw new Error('Matrices are not "dot" compatible!');
    }
    let m = new Matrix(m0.rows, m1.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        let sum = 0;
        for (let k = 0; k < m0.cols; k++) {
          sum += m0.data[i][k] * m1.data[k][j];
        }
        m.data[i][j] = sum;
      }
    }
    return m;
  }

  // Aplicar una función en cada celda de una matriz
  static map(m0, mFunction) {
    let m = new Matrix(m0.rows, m0.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        m.data[i][j] = mFunction(m0.data[i][j]);
      }
    }
    return m;
  }

  // Multiplicar dos matrices
  static multiply(m0, m1) {
    Matrix.checkDimensions(m0, m1);
    let m = new Matrix(m0.rows, m0.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        m.data[i][j] = m0.data[i][j] * m1.data[i][j];
      }
    }
    return m;
  }

  // Restar dos matrices
  static subtract(m0, m1) {
    Matrix.checkDimensions(m0, m1);
    let m = new Matrix(m0.rows, m0.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        m.data[i][j] = m0.data[i][j] - m1.data[i][j];
      }
    }
    return m;
  }

  // Transponer una matriz
  static transpose(m0) {
    let m = new Matrix(m0.cols, m0.rows);
    for (let i = 0; i < m0.rows; i++) {
      for (let j = 0; j < m0.cols; j++) {
        m.data[j][i] = m0.data[i][j];
      }
    }
    return m;
  }

  // Aplicar pesos aleatorios entre -1 y 1
  randomWeights() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }
}
