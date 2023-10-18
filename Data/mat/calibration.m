clear all; 
close all;
clc;

%I measured 'precise'-ish distances- Thanos to wall at 50cm, 60cm, ... 70cm
%then I took the average output of the 4 center pixels from the depth
%values output from thanos_phase_one

%the pgm values in the 1000s are what i got from setting the
%thanos at the corresponding dists, did a polyfit and the equation is below
%(the line where y = mx + b) this equation TRANSLATES a pgm pixel value to
%a pure cm distance- check the units on the cm distance just in case its
%not like mm or something but im fairly certain the eq is converting to cm
dist = [50 60 70 80 90 100];
pgm  = [4144 4897 5697 6521 7331 8089];

plot(dist, pgm,'-o')
j = polyfit(dist,pgm,1);
x = linspace(50,100);
y = j(2) + j(1).*x;
hold on
plot(x,y)

f = polyval(j,dist);
T = table(dist,pgm,f,pgm-f,'VariableNames', {'dist','pgm','Fit','FitError'})
