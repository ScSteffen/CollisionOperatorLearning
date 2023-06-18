using Flux, NeuralOperators

branch = Chain(Dense(32, 64, sigmoid), Dense(64, 72, sigmoid))
trunk = Chain(Dense(24, 64, tanh), Dense(64, 72, tanh))
model = DeepONet(branch, trunk)

model(rand(32), rand(24))
