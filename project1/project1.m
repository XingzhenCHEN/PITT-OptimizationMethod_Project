
f = [zeros(16, 1); ones(5, 1)];
% first part of constraints
Aeq1 = [ones(4, 1); zeros(17, 1);];

Aeq2 = -[-ones(4, 1); ones(4, 1); zeros(13, 1)];

Aeq3 = [zeros(4, 1); -ones(4, 1); ones(4, 1); zeros(9, 1)];

Aeq4 = -[zeros(8, 1); -ones(4, 1); ones(4, 1); zeros(5, 1)];

Aeq5 = [zeros(12, 1); -ones(4, 1); zeros(5, 1)];

Aeq = [Aeq1'; Aeq2'; Aeq3'; Aeq4'; Aeq5'];
beq = [2 1 2 1 -2]';

% second part of constraints
T = [1 2 5 10];
A1 = {1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0;
      0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0;
      0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0;
      0 0 0 10 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0;
      1 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0;
      0 2 0 0 0 -2 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0;
      0 0 5 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 -1 0 0 0;
      0 0 0 10 0 0 0 -10 0 0 0 0 0 0 0 0 0 -1 0 0 0;
      0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 -1 0 0;
      0 0 0 0 0 -2 0 0 0 2 0 0 0 0 0 0 0 0 -1 0 0;
      0 0 0 0 0 0 -5 0 0 0 5 0 0 0 0 0 0 0 -1 0 0;
      0 0 0 0 0 0 0 -10 0 0 0 10 0 0 0 0 0 0 -1 0 0;
      0 0 0 0 0 0 0 0 1 0 0 0 -1 0 0 0 0 0 0 -1 0;
      0 0 0 0 0 0 0 0 0 2 0 0 0 -2 0 0 0 0 0 -1 0;
      0 0 0 0 0 0 0 0 0 0 5 0 0 0 -5 0 0 0 0 -1 0;
      0 0 0 0 0 0 0 0 0 0 0 10 0 0 0 -10 0 0 0 -1 0;
      0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 -1;
      0 0 0 0 0 0 0 0 0 0 0 0 0 -2 0 0 0 0 0 0 -1;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -1;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -10 0 0 0 0 -1;};




% third part of constraints
A2 = {-1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      -1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 1 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0  1 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 1 0 0 0 -1 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 1 0 0 0 -1 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 1 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0;};


A = [A1; A2];
A = cell2mat(A);

b = {zeros(16,1);-1;-2;-5;-10;zeros(16,1);1;1;1;1;};
b = cell2mat(b);


% fourth part of constraints
lb = zeros(21, 1);
ub = [ones(16, 1); Inf * ones(5, 1)];
intcon = 1 : 21;

% solve the integer LP
x = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub);

% interpret the solution
P1 = x(1 : 4)';
P2 = x(5 : 8)';
P3 = x(9 : 12)';
P4 = x(13 : 16)';
P5 = {1;1;1;1;};
P5 = cell2mat(P5);



S1 = P1;
S2 = P2 - P1;
S3 = P3 - P2;
S4 = P4 - P3;
S5 = P5 - P4;

% transition matrix S
S = int32([S1; S2; S3; S4; S5]);

disp(S)

 