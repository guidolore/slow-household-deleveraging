% Optimal consumption and saving problem with housing,
% with adjustment costs and long-term mortgages
%
% Solves steady state of model in Guerrieri-Lorenzoni-Prato (JEEA, 2020)
%
% Uses LCP.m to solve linear complementarity problem
%
% September 2020

clear
close all
clc

load_ind = 1;       % set to 1 to load previously saved values          

for i_ss = 1:2
    % i_ss = 1 computes and save steady state before shock
    % i_ss = 2 computes and save steady state after shock

if i_ss == 1
    p = 1;                  % price of housing
else
    p = 0.9;
end

% parameters
rho         = 0.06;     % discount rate
sigma       = 2;        % coefficient of relative risk aversion
alpha       = 0.85;     % share of non-housing consumption
delta       = 0.02;     % housing depreciation
r           = 0.03;     % interest rate
phi         = 0.04;     % transaction cost
theta       = 0.8;      % LTV constraint
tau         = 30;       % mortgage term
Dt          = .1;       % time period (only used for stationary distribution)
Delta       = 1000;

% income process
I_y = 2;
y = [.1 .2];
lambda = [.1; .1];
g_y = [lambda(2), lambda(1)]/sum(lambda);
Y = g_y*y';

% grid for liquid wealth a 
I_a = 60;
a_min = 0;
a_max = 3;
a = linspace(a_min,a_max,I_a)';
Da = a(2)-a(1);

% grid for housing h
I_h = 70;
h_min = 0.2*(1-alpha)*y(1)/(r+delta);
h_max = 1.2*(1-alpha)*y(2)/(r+delta);
h = linspace(h_min,h_max,I_h);

% mortgage payment m
m = r/(1-exp(-r*tau))*p*theta*h;

% grid for mortgage debt d
% the grid is different for different levels of h
I_d = 50;
d = zeros(I_d,I_h);
Dd = zeros(1,I_h);
for i_h=1:I_h
d(:,i_h) = linspace(0,theta*h(i_h),I_d); 
Dd(i_h) = d(2,i_h)-d(1,i_h);
end

% grid for net wealth
I_w = 80;
% w_min is the minimum value of wealth at which adjustment is feasible
w_min = (1-theta)*p*h_min;
w_max = max(p*h)+max(a);
w = linspace(w_min,w_max,I_w);
Dw = w(2)-w(1);

% total number of states 
N = I_a*I_d*I_y*I_h;
% total number of states for given h
N1 = I_a*I_d*I_y; 

% build matrices of state variables
% these are I_a x I_d x I_y x I_h
aa = repmat(a,1,I_d,I_y,I_h);
dd = permute(repmat(d,1,1,I_a,I_y),[3 1 4 2]);
yy = permute(repmat(y',1,I_a,I_d,I_h),[2 3 1 4]);
hh = permute(repmat(h',1,I_a,I_d,I_y),[2 3 4 1]);

% this one is I_a x I_d for HJB at constant h
aa1 = repmat(a,1,I_d);

% these are I_a x I_d x I_h for maximization at adjustment
aa2 = repmat(a,1,I_d,I_h);
dd2 = permute(repmat(d,1,1,I_a),[3 1 2]);
hh2 = permute(repmat(h',1,I_a,I_d),[2 3 1]);

% net wealth at adjustment
ww = aa2 + (1-phi)*p*hh2 - dd2;
ww = max(w(1),min(w(end),ww));

% allocate cell of transition matrices 
A = cell(1,I_h);
AT = cell(1,I_h);

% create matrix with exogenous transitions for d and y

% transition for y
A_y = [-lambda(1),lambda(1);lambda(2) -lambda(2)];
A_y = kron(A_y,speye(I_a*I_d));

% transition for d
A_yd = cell(1,I_h);
for i_h=1:I_h
    d_dot = min(r*d(:,i_h) - m(i_h),0);
    d_dot(1) = 0;
    A_d = 1/Dd(i_h)*spdiags([d_dot(1:I_d), -d_dot(1:I_d)],[0 1],I_d,I_d)';
    A_d = kron(A_d,speye(I_a));
    A_d = kron(speye(I_y),A_d);      
    A_yd{i_h} = A_d + A_y;    
end

% preallocate some matrices
sF = zeros(I_a,I_d,I_y);
sB = zeros(I_a,I_d,I_y);
Vi = zeros(I_a,I_d,I_y,I_h);
J = zeros(I_w,I_y);
polJ = zeros(I_w,I_y);
v_try = zeros(I_w,I_h);

%% FIND VALUE FUNCTION
disp('Compute initial value for V')

% COMPUTE VALUE FUNCTION WITH NO ADJUSTMENT TO USE AS INITIAL CONDITION

if load_ind == 1
    if i_ss == 1
        load steady_state0 
        V = V0;
    else
        load steady_state1
        V = V1;
    end
else
    
% initial guess
V = 1/rho*((max(r*(aa-dd)+yy-delta*p*hh,.0001)).^alpha.*(hh).^(1-alpha)).^(1-sigma)/(1-sigma);

% value function iterations
for i_h = 1:I_h
v = V(:,:,:,i_h);
z = r*a + yy(:,:,:,i_h) - m(i_h) - delta*p*h(i_h);

crit = 1;
while crit>1e-8
    
% compute dV/da
Dv_a = (v(2:I_a,:,:)-v(1:I_a-1,:,:))/Da;

% optimal c
RHS = Dv_a./(alpha.*(h(i_h).^((1-alpha)*(1-sigma))));
c   = RHS.^(1/(alpha*(1-sigma)-1));

% compute saving rates going forward and backward
sF(1:I_a-1,:,:) = z(1:I_a-1,:,:) - c;
sB(2:I_a,:,:) = z(2:I_a,:,:) - c;
sF(I_a,:,:) = 0;
sB(1,:,:) = 0;

% choose side using upwind scheme
cF = z - sF;
cB = z - sB;
obj_F = (cF.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma) + sF.*[Dv_a;zeros(1,I_d,I_y)];
obj_B = (cB.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma) + sB.*[zeros(1,I_d,I_y);Dv_a];
i1 = sF>0 & sB>=0;
i2 = sB<0 & sF<=0;
i3 = sF>0 & sB<0 & obj_F>obj_B;
i4 = sF>0 & sB<0 & obj_F<obj_B;

s = zeros(I_a,I_d,I_y);
s(i1|i3) = sF(i1|i3);
s(i2|i4) = sB(i2|i4);

c = z - s;
u = (c.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma);

% form matrix A adding exogenous transitions for d and y
s = s(:)/Da;
A{i_h} = A_yd{i_h} + sparse(1:N1,1:N1,-max(s,0)+min(s,0)) + ...
        sparse(2:N1,1:N1-1,-min(s(2:N1),0),N1,N1) + ...
            sparse(1:N1-1,2:N1,max(s(1:N1-1),0),N1,N1);

% solve for new values vi using implicit method
B = (rho+1/Delta)*speye(N1)-A{i_h};
b = u(:) + 1/Delta*v(:);
vi = B\b;
crit = max(abs(vi-v(:)));
v = reshape(vi,I_a,I_d,I_y);
end

% update values
V(:,:,:,i_h) = v;
end
end

%% FIND VALUE FUNCTION 
disp('Value function iterations')

crit = 1;
while crit>1e-8
  
% SOLVE OPTIMIZATION PROBLEM AT ADJUSTMENT
for i_y = 1:I_y
for i_h = 1:I_h     % candidate new asset positions
    dd1 = dd2(:,:,i_h);
    V_interp = griddedInterpolant(aa1,dd1,V(:,:,i_y,i_h));
    ai = w' - (1-theta)*p*h(i_h);
    i_a_neg = ai<0;
    ai = min(ai,a_max);
    v_try(:,i_h) = V_interp(ai,ones(I_w,1)*theta*p*h(i_h));
    v_try(i_a_neg,i_h) = -inf;
end
% stores values in J and optimal policy in polJ
    [J(:,i_y), polJ(:,i_y)] = max(v_try,[],2);
end

% interpolate at state variables a d h
V_adj = zeros(I_a,I_d,I_h,I_y);
for i_y = 1:I_y
Vmax = interp1(w,J(:,i_y),ww);
V_adj(:,:,:,i_y) = Vmax; 
end
V_adj = permute(V_adj,[1 2 4 3]);

% set to low value states where adjustment is not possible
V_adj(aa-dd+(1-phi)*p*hh<(1-theta)*p*h_min) = -1e4;

% SOLVE HJBVI USING IMPLICIT METHOD AND LCP ALGORITHM
for i_h = 1:I_h
v = V(:,:,:,i_h);
z = r*a + yy(:,:,:,i_h) - m(i_h) - delta*p*h(i_h);

% compute dV/da
Dv_a = (v(2:I_a,:,:)-v(1:I_a-1,:,:))/Da;

% optimal c
RHS = Dv_a./(alpha.*(h(i_h).^((1-alpha)*(1-sigma))));
c   = RHS.^(1/(alpha*(1-sigma)-1));

% compute saving rates going forward and backward
sF(1:I_a-1,:,:) = z(1:I_a-1,:,:) - c;
sB(2:I_a,:,:) = z(2:I_a,:,:) - c;
sF(I_a,:,:) = 0;
sB(1,:,:) = 0;

% choose side using upwind scheme
cF = z - sF;
cB = z - sB;
obj_F = (cF.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma) + sF.*[Dv_a;zeros(1,I_d,I_y)];
obj_B = (cB.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma) + sB.*[zeros(1,I_d,I_y);Dv_a];
i1 = sF>0 & sB>=0;
i2 = sB<0 & sF<=0;
i3 = sF>0 & sB<0 & obj_F>obj_B;
i4 = sF>0 & sB<0 & obj_F<obj_B;

s = zeros(I_a,I_d,I_y);
s(i1|i3) = sF(i1|i3);
s(i2|i4) = sB(i2|i4);

c = z - s;
u = (c.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma);

% form matrix A adding exogenous transitions for d and y
s = s(:)/Da;
A{i_h} = A_yd{i_h} + sparse(1:N1,1:N1,-max(s,0)+min(s,0)) + ...
        sparse(2:N1,1:N1-1,-min(s(2:N1),0),N1,N1) + ...
            sparse(1:N1-1,2:N1,max(s(1:N1-1),0),N1,N1);

% optimal values at stopping        
Vstar = V_adj(:,:,:,i_h);

% form matrices and vectors for LCP
B = (rho+1/Delta)*speye(N1) - A{i_h};
b = u(:) + 1/Delta*v(:);
q = - b + B*Vstar(:); 
x0 = v(:) - Vstar(:);
% solve LCP 
x = LCP(B,q,zeros(N1,1),inf*ones(N1,1),x0,0);
v = Vstar(:) + x;
v = reshape(v,I_a,I_d,I_y);

% update values
Vi(:,:,:,i_h) = v;
end

% check convergence and update V
crit = max(abs(V(:)-Vi(:)));
disp(crit)
V = Vi;

end

%% COMPUTE OPTIMAL POLICY FOR CONSUMPTION AND SAVING
c_pol = zeros(I_a,I_d,I_y,I_h);
s_pol = zeros(I_a,I_d,I_y,I_h);

for i_h=1:I_h

v = V(:,:,:,i_h);
z = r*a + yy(:,:,:,i_h) - m(i_h) - delta*p*h(i_h);

% compute dV/da
Dv_a = (v(2:I_a,:,:)-v(1:I_a-1,:,:))/Da;

% optimal c
RHS = Dv_a./(alpha.*(h(i_h).^((1-alpha)*(1-sigma))));
c   = RHS.^(1/(alpha*(1-sigma)-1));

% compute saving rates going forward and backward
sF(1:I_a-1,:,:) = z(1:I_a-1,:,:) - c;
sB(2:I_a,:,:) = z(2:I_a,:,:) - c;
sF(I_a,:,:) = 0;
sB(1,:,:) = 0;

% choose side using upwind scheme
cF = z - sF;
cB = z - sB;
obj_F = (cF.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma) + sF.*[Dv_a;zeros(1,I_d,I_y)];
obj_B = (cB.^alpha.*h(i_h).^(1-alpha)).^(1-sigma)/(1-sigma) + sB.*[zeros(1,I_d,I_y);Dv_a];
i1 = sF>0 & sB>=0;
i2 = sB<0 & sF<=0;
i3 = sF>0 & sB<0 & obj_F>obj_B;
i4 = sF>0 & sB<0 & obj_F<obj_B;

s = zeros(I_a,I_d,I_y);
s(i1|i3) = sF(i1|i3);
s(i2|i4) = sB(i2|i4);

c_pol(:,:,:,i_h) = z - s;
s_pol(:,:,:,i_h) = s;
end

%% COMPUTE STATIONARY DISTRIBUTION
disp('Compute stationary distribution')
% BUILD MATRIX M FOR ADJUSTMENT DYNAMICS

% find stopping states and optimal policy at adjustment
i_stop = V==V_adj; % indicator of stopping

% new housing and debt levels
ii_h = zeros(I_a,I_d,I_y,I_h);
ii_d = zeros(I_a,I_d,I_y,I_h);
i_ww = 1 + floor((ww-w(1))/Dw);
for i_y=1:I_y
    % use polVV to set new housing index ii_h
    ii_h(:,:,i_y,:) = reshape(polJ(i_ww(:),i_y),I_a,I_d,1,I_h);
end
for i_h=1:I_h
    % new debt index ii_d
    ii_d(:,:,:,i_h) = min(1 + floor((theta*p*h(ii_h(:,:,:,i_h)))./Dd(ii_h(:,:,:,i_h))),I_d);
end
% new liquid asset level
ai = aa + (1-phi)*p*hh - dd - (1-theta)*p*h(ii_h);
ai = max(min(ai,a_max),a_min);
ii_a = min(1 + floor((ai-a(1))/Da),I_a-1);
weigh = (ai-a(ii_a))/Da;        
ii_y = permute(repmat((1:I_y)',1,I_a,I_d,I_h),[2 3 1 4]);
% index of new state when adjusting
i_move = ii_a + (ii_d-1)*I_a + (ii_y-1)*I_a*I_d + (ii_h-1)*I_a*I_d*I_y;

% indexes to build matrix M
nn = 1:N;
nno = nn(~i_stop);
nns = nn(i_stop);
i_mo = i_move(i_stop);
wei = weigh(i_stop);

% build matrix M
M = sparse(nns,i_mo,1-wei,N,N) + sparse(nns,i_mo+1,wei,N,N) + sparse(nno,nno,1,N,N);
MT = transpose(M);

% transpose A
for i_h=1:I_h
    AT{i_h} = transpose(A{i_h});
end

% initial value for distribution

if load_ind == 1
    if i_ss == 1
        g = gss_0;
    else
        g = gss_1;
    end
else
g = zeros(N,1);
g(1) = 1;
end
gi = zeros(N,1);

% ITERATE FORWARD TO FIND STATIONARY DISTRIBUTION g

crit = 1;
while crit>1e-10
    for i_h=1:I_h
        ii = (1:N1) + (i_h-1)*N1;
        gi(ii) = g(ii) + AT{i_h}*g(ii)*Dt;
    end
    gi = MT*gi;
    crit = max(abs(gi-g));
    disp(crit)
    g = gi;
end

% COMPUTE MARGINALS AND STEADY STATE MOMENTS

g = reshape(g,[I_a I_d I_y I_h]);
g_a = sum(sum(sum(g,4),3),2)';
g_d = reshape(sum(sum(g,1),3),I_d,I_h);
g_h = reshape(sum(sum(sum(g,3),2),1),[1 I_h]);

C_ss = g(:)'*c_pol(:);
D_ss = g(:)'*dd(:);
A_ss = g(:)'*aa(:);
H_ss = g(:)'*hh(:);

% compute adjustment rate and adjustment costs
gi = g(:);
for i_h=1:I_h
    ii = (1:N1) +(i_h-1)*N1;
    gi(ii) = gi(ii) + AT{i_h}*gi(ii)*Dt;
end
adj_rate = gi'*i_stop(:)/Dt;
adj_cost = (i_stop(:).*gi(:))'*phi*hh(:)/Dt;

% compute conditional adjustment rate for agents who switch from y2 to y1
gi = g;
gi(:,:,1,:) = gi(:,:,2,:); % move mass of agents at y2 to y1
gi(:,:,2,:) = 0;
gi = gi(:)/sum(gi(:)); % conditional distribution of agents who just switched y2->y1
trading_y2y1 = gi'*i_stop(:)/Dt; % trading rate

% compute MPC
MPCg = (c_pol(2:I_a,:,:,:)-c_pol(1:I_a-1,:,:,:)).*g(1:I_a-1,:,:,:)/Da;
MPC = sum(MPCg(:));

% median wealth
nw = aa+(1-phi)*hh-dd;
dat = [nw(:) g(:)];
dat = sortrows(dat,1);
dat(:,2) = cumsum(dat(:,2));
i_median = find(dat(:,2)>.5,1);
nw_median = dat(i_median,1);

% save parameters and grids
save params rho sigma alpha delta r phi theta tau lambda Delta Dt
save grids a I_a Da a_min a_max d I_d Dd y I_y h I_h h_min m w I_w Dw N N1 ...
    aa aa1 aa2 dd dd2 yy hh hh2 ww A_yd
    
% save values, policies and moments for transitional dynamics
if i_ss == 1
    V0 = V;
    gss_0 = g(:);
    save steady_state0 V0 gss_0 C_ss A_ss D_ss H_ss Y
else
    V1 = V;
    V_adj1 = V_adj;
    gss_1 = g(:);
    save steady_state1 V1 V_adj1 gss_1 MT AT c_pol i_stop i_move weigh
end

%% PLOTS

figure(1)
S_pol=s_pol(:,10,1,1);
Ind = i_stop(:,10,1,1)==0;
plot(a(Ind),S_pol(Ind),'LineWidth',2)
hold on
S_pol=s_pol(:,I_d,1,1);
Ind = i_stop(:,I_d,1,1)==0;
plot(a(Ind),S_pol(Ind),'LineWidth',2)
S_pol=s_pol(:,I_d,I_y,I_h);
Ind = i_stop(:,I_d,I_y,I_h)==0;
plot(a(Ind),S_pol(Ind),'LineWidth',2)
S_pol=s_pol(:,I_d-5,I_y,I_h);
Ind = i_stop(:,I_d-5,I_y,I_h)==0;
plot(a(Ind),S_pol(Ind),'LineWidth',2)
title('saving policies')

figure(2)
C_pol=c_pol(:,I_d,I_y,I_h);
Ind = i_stop(:,I_d,I_y,I_h)==0;
plot(a(Ind),C_pol(Ind),'LineWidth',2)
hold on

figure(3)
plot(a,g_a,'LineWidth',2)
title('distribution of liquid assets')
hold on

figure(4)
plot(d,g_d,'LineWidth',2)
hold on

figure(5)
plot(h,g_h,'LineWidth',2)
hold on
end
