function [energy] = energy(signal)

energy = sum(abs(signal).^2);