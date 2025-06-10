% Optimal consumption and saving problem with housing,
% with adjustment costs and long-term mortgages
%
% Solves transitional dynamics after a house price shock
% of model in Guerrieri-Lorenzoni-Prato (JEEA, 2020)
%
% Run after computing steady state with illiq_hous.m
% Uses LCP.m to solve linear complementarity problem
%
% September 2020

clear
close all
clc

T = 500;
p = 0.9;

load params
load grids
load steady_state1

V = V1;
V_adj = V_adj1;

% preallocate some matrices
sF = zeros(I_a,I_d,I_y);
sB = zeros(I_a,I_d,I_y);
Vi = zeros(I_a,I_d,I_y,I_h);

%% FIND VALUE FUNCTION FOR CONSUMERS WITH PRE-SHOCK MORTGAGES

% mortgage payment
p0 = 1;
m = r/(1-exp(-r*tau))*p0*theta*h;

% allocate cell of transition matrices 
Am = cell(1,I_h);

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

%% VALUE FUNCTION ITERATIONS FOR OLD MORTGAGES

disp('Value function iterations for old mortgages')

crit = 1;
while crit>1e-8
  
%% SOLVE HJBVI USING IMPLICIT METHOD AND LCP ALGORITHM

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

% form matrix Am adding exogenous transitions for d and y
s = s(:)/Da;
Am{i_h} = A_yd{i_h} + sparse(1:N1,1:N1,-max(s,0)+min(s,0)) + ...
        sparse(2:N1,1:N1-1,-min(s(2:N1),0),N1,N1) + ...
            sparse(1:N1-1,2:N1,max(s(1:N1-1),0),N1,N1);

% optimal values at stopping        
Vstar = V_adj(:,:,:,i_h);

% form matrices and vectors for LCP
B = (rho+1/Delta)*speye(N1) - Am{i_h};
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

% check convergence
crit = max(abs(V(:)-Vi(:)));
disp(crit)
V = Vi;
end

% save values 
Vm = V;
A_ydm = A_yd;
save old_mort Vm A_ydm

% COMPUTE OPTIMAL POLICY FOR CONSUMPTION AND SAVING
c_pol0 = zeros(I_a,I_d,I_y,I_h);
s_pol0 = zeros(I_a,I_d,I_y,I_h);

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

c_pol0(:,:,:,i_h) = z - s;
s_pol0(:,:,:,i_h) = s;
end

% BUILD MATRIX M FOR ADJUSTMENT DYNAMICS

% find stopping states and optimal policy at adjustment
% i_move and i_weigh come from the optimal adjustment in steady state 1
i_stop0 = V==V_adj; % indicator of stopping

% indexes to build matrix M
nn = 1:N;
nno = nn(i_stop0==0);
nns = nn(i_stop0);
i_mo = i_move(i_stop0);
wei = weigh(i_stop0);

% agents who do not adjust remain with old mortgage
Ma = sparse(nno,nno,1,N,N); 
% agents who adjust switch to new mortgage distribution
Mb = sparse(nns,i_mo,1-wei,N,N) + sparse(nns,i_mo+1,wei,N,N);

MaT = transpose(Ma);
MbT = transpose(Mb);

AmT = cell(1,I_h);
for i_h=1:I_h
    AmT{i_h} = transpose(Am{i_h});
end

%% iterations of M and A*Dt

% agents with all mortgages are in g0
% agents with new mortgages are in g1

% initial distribution
load steady_state0
g0 = gss_0; 
g0i = zeros(N,1); 

load steady_state1
g1 = zeros(N,1); 
g1i = zeros(N,1); 

Cs = zeros(1,T); 
As = zeros(1,T);
Ds = zeros(1,T);
Hs = zeros(1,T);
Sg0 = zeros(1,T);

Cs(1) = C_ss;
As(1) = (g0(:)+g1(:))'*aa(:);
Ds(1) = (g0(:)+g1(:))'*dd(:);
Hs(1) = (g0(:)+g1(:))'*hh(:);
Sg0(1) = sum(g0(:));

for t = 2:T
    disp(t)
    Cs(t) = c_pol(:)'*g1 + c_pol0(:)'*g0;
    for i_h=1:I_h
        ii = (1:N1) + (i_h-1)*N1;
        g0i(ii) = g0(ii) + AmT{i_h}*g0(ii)*Dt;
        g1i(ii) = g1(ii) + AT{i_h}*g1(ii)*Dt;
    end
    g0 = MaT*g0i;  
    g1 = MT*g1i + MbT*g0i;
    As(t) = (g0(:)+g1(:))'*aa(:);
    Ds(t) = (g0(:)+g1(:))'*dd(:);
    Hs(t) = (g0(:)+g1(:))'*hh(:);
    Sg0(t) = sum(g0(:));
end

%% PLOTS

tt = (-100:T)*Dt;
figure(6)
subplot(4,1,1)
plot(tt,[zeros(1,101), 100*(Cs/Cs(1)-1)],'LineWidth',2)
hold on
subplot(4,1,2)
plot(tt,[zeros(1,101), Ds./(p*Hs)-D_ss/H_ss],'LineWidth',2)
hold on
subplot(4,1,3)
plot(tt,[ones(1,101), (As-Ds+(1-phi)*p*Hs)/(A_ss-D_ss+(1-phi)*H_ss)],'LineWidth',2)
hold on
subplot(4,1,4)
plot(tt,[ones(1,101), Sg0],'LineWidth',2)
hold on

save cons_TD tt Cs As Ds Hs A_ss D_ss H_ss Sg0
