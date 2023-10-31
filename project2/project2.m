% Steepest Descent

syms w a0 a1 a2
N = 5;
w = [0.5236; 1.0472; 1.5708; 2.0944; 2.6180];
Pw = [0.6821; 4.3232; 3.7540; 0.4368; 0.1988];
E(a0, a1, a2) = sym(0);
for i = 1 : 5
    Pwi_hat(a0, a1, a2) = (a0 + a1 * exp(-1j * w(i)) + a2 * exp(-2j * w(i))) * conj(a0 + a1 * exp(-1j * w(i)) + a2 * exp(-2j * w(i)));
    tmp = Pw(i) * Pwi_hat - log(Pw(i) * Pwi_hat) - 1;
    E(a0, a1, a2) = E(a0, a1, a2) + tmp;
end
dE0 = vpa(diff(E, a0), 10);
dE1 = vpa(diff(E, a1), 10);
dE2 = vpa(diff(E, a2), 10);

step = Inf;
a = [1 0 0]';
E_record = [];
distance_record = [];
optimal = [1; -0.5161; 0.9940];
count = 0;
lr = 1;

% Begin
while step > 1e-6
    
    E_begin = double(E(a(1), a(2), a(3)));
    ga0 = double(dE0(a(1), a(2), a(3)));
    ga1 = double(dE1(a(1), a(2), a(3)));
    ga2 = double(dE2(a(1), a(2), a(3)));
    

    grad = [ga0; ga1; ga2];
    p = -grad;
    p0 = -ga0;
    p1 = -ga1;
    p2 = -ga2;
    while double(E(a(1) + lr * p0, a(2) + lr * p1, a(3) + lr * p2)) > double(E(a(1), a(2), a(3)) + lr * 0.5 * grad' * p)
        lr = 0.95 * lr;
    end
    
    a(1) = a(1) + lr * p0;
    a(2) = a(2) + lr * p1;
    a(3) = a(3) + lr * p2;
    
    step = abs(double(E(a(1), a(2), a(3))) - E_begin);
    
    E_record = [E_record; E_begin];
    distance_record = [distance_record; (a - optimal)' * (a - optimal)];
    
    count = count + 1;
    disp(a)
    
end

figure(1)
semilogy((1 : length(E_record)), real(E_record));
title('E vs Iteration');
xlabel('iterations');
ylabel('E value');

figure(2)
semilogy((1 : length(distance_record)), real(distance_record));
title('distance to optimal solution vs Iteration');
xlabel('iteration');
ylabel('distance');
