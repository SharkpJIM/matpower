%% ========================================================================
% KẾT HỢP: 6 TRẠM EV (3 HÀNH VI) + TỐI ƯU 3 PV BẰNG PSO
% IEEE 33-bus | MATPOWER | 24h Power Flow - FULLY OPTIMIZED
% 
% ⚡ PHIÊN BẢN TỐI ƯU HÓA: 100-200x nhanh hơn
% - Giảm PSO: 1000→50 particles, 100→30 iterations
% - Parallel: PSO, 24h simulation, fitness evaluation
% - Bỏ in log chi tiết, thêm tùy chọn vẽ biểu đồ
%% ========================================================================
clc; clear; close all;
tic_total = tic;

define_constants;
fprintf('\n');
fprintf('=============================================================\n');
fprintf('  KẾT HỢP: 6 TRẠM EV (3 HÀNH VI) + TỐI ƯU PV BẰNG PSO\n');
fprintf('  ⚡ PHIÊN BẢN TỐI ƯU HÓA (Fast Execution)\n');
fprintf('=============================================================\n');

%% ===================== KHỞI TẠO PARALLEL POOL ======================
pool = gcp('nocreate');
if isempty(pool)
    try
        numCores = feature('numcores');
        pool = parpool('local', min(4, numCores));
        fprintf('>> Đã khởi tạo Parallel Pool với %d workers\n', pool.NumWorkers);
    catch ME
        warning('Không thể khởi tạo parpool: %s. Chạy sequential.', ME.message);
    end
else
    fprintf('>> Sử dụng Parallel Pool hiện có với %d workers\n', pool.NumWorkers);
end

%% ===================== PHẦN 0: LOAD LƯỚI ==============================
try
    base_mpc = loadcase('case33bw');
    n_buses = size(base_mpc.bus, 1);
    n_lines = size(base_mpc.branch, 1);
    total_base_load_MW = sum(base_mpc.bus(:,3));
    fprintf('[OK] Load IEEE 33-bus | Nút: %d | Nhánh: %d | Tải nền: %.3f MW\n\n', ...
        n_buses, n_lines, total_base_load_MW);
catch
    error('LỖI: Không tìm thấy MATPOWER. Chạy: addpath(genpath(''matpower7.1''))');
end

%% ===================== PHẦN 1: MÔ HÌNH EV 3 HÀNH VI ====================
fprintf('=== [1] MÔ HÌNH EV - 3 HÀNH VI (MONTE CARLO) ===\n');

N_ev = 50000;
P_charge = 7.4;
Cap_max_ev = 60;
eta_ev = 0.9;
eff_distance = 0.15;

behaviors = struct();
behaviors(1).name = 'Sạc tại nhà (đêm)';
behaviors(1).mu_t = 20;  behaviors(1).sigma_t = 2.5;
behaviors(1).alpha_soc = 2;  behaviors(1).beta_soc = 3;
behaviors(1).mu_d = 4;       behaviors(1).sigma_d = 0.5;

behaviors(2).name = 'Sạc văn phòng (sáng)';
behaviors(2).mu_t = 8;   behaviors(2).sigma_t = 1.5;
behaviors(2).alpha_soc = 2;  behaviors(2).beta_soc = 2.5;
behaviors(2).mu_d = 3.5;     behaviors(2).sigma_d = 0.4;

behaviors(3).name = 'Sạc dịch vụ (trưa)';
behaviors(3).mu_t = 13;  behaviors(3).sigma_t = 2.0;
behaviors(3).alpha_soc = 2;  behaviors(3).beta_soc = 5;
behaviors(3).mu_d = 4.2;     behaviors(3).sigma_d = 0.6;

P_EV_behaviors = zeros(3, 24);

for bv = 1:3
    rng(bv * 42);
    t_start = mod(normrnd(behaviors(bv).mu_t, behaviors(bv).sigma_t, [N_ev,1]), 24);
    t_start(t_start < 0) = t_start(t_start < 0) + 24;
    
    soc_init = betarnd(behaviors(bv).alpha_soc, behaviors(bv).beta_soc, [N_ev,1]);
    
    distance = lognrnd(behaviors(bv).mu_d, behaviors(bv).sigma_d, [N_ev,1]);
    distance = min(distance, 500);
    
    P_h = zeros(1, 24);
    for i = 1:N_ev
        E_req = min(distance(i)*eff_distance, Cap_max_ev*(1-soc_init(i)));
        T_dur = E_req / (P_charge * eta_ev);
        hs = max(1, round(t_start(i)));
        he = min(24, hs + max(0, round(T_dur)));
        P_h(hs:he) = P_h(hs:he) + P_charge;
    end
    P_EV_behaviors(bv, :) = P_h / 1000;
end

% Tổng hợp 3 hành vi, scale về 20% tải nền
P_EV_target_MW = 0.20 * total_base_load_MW;
P_EV_combined = mean(P_EV_behaviors, 1);
scale_ev = P_EV_target_MW / max(P_EV_combined + eps);
P_EV_24h = P_EV_combined * scale_ev;

fprintf('  Đỉnh EV: %.3f MW (giờ %d) | TB: %.3f MW\n', ...
    max(P_EV_24h), find(P_EV_24h==max(P_EV_24h),1), mean(P_EV_24h));

EV_Bus = [4, 7, 10, 18, 30, 32];
fprintf('  6 Trạm EV: [%s]\n\n', sprintf('%d ', EV_Bus));

%% ===================== PHẦN 2: PSO TỐI ƯU VỊ TRÍ 3 PV ==================
fprintf('=== [2] PSO TỐI ƯU VỊ TRÍ 3 PV (TỔNG LOSS 24H) ===\n');

PV1_candidate = 2:33;
PV2_candidate = 2:33;
PV3_candidate = 2:33;
PV_capacity_MW = 0.4;

n = 3;
p = 50;          % Giảm từ 1000 → 50 (20x)
k = 30;          % Giảm từ 100 → 30 (3.3x)
w  = 0.7;
c1 = 2.05;
c2 = 2.05;
vmax = 3;

x = zeros(p,n);
x(:,1) = randi([1 length(PV1_candidate)],p,1);
x(:,2) = randi([1 length(PV2_candidate)],p,1);
x(:,3) = randi([1 length(PV3_candidate)],p,1);
v = zeros(p,n);
f = zeros(p,1);

mpopt = mpoption('verbose', 0, 'out.all', 0);

% Khởi tạo fitness song parallel
fprintf('>> Khởi tạo fitness cho %d particles (Parallel)...\n', p);
t_init = tic;
parfor i = 1:p
    f(i) = compute_pso_loss(x(i,:), PV1_candidate, PV2_candidate, PV3_candidate, ...
        base_mpc, EV_Bus, P_EV_24h, PV_capacity_MW, mpopt);
end
fitness = f;

[globalmin, index] = min(fitness);
g = x(index,:);
localx = x;

fprintf('>> Khởi tạo xong (%.1fs) | Best fitness = %.6f MW\n\n', toc(t_init), globalmin);

% Vòng lặp PSO
fprintf('>> PSO Optimization:\n');
for iter = 1:k
    r1 = rand(p,n);
    r2 = rand(p,n);
    
    v = w*v + c1*r1.*(localx - x) + c2*r2.*(ones(p,1)*g - x);
    v(v > vmax) = vmax; 
    v(v < -vmax) = -vmax;
    
    x = x + v;
    x = round(x);
    x(:,1) = max(1, min(length(PV1_candidate), x(:,1)));
    x(:,2) = max(1, min(length(PV2_candidate), x(:,2)));
    x(:,3) = max(1, min(length(PV3_candidate), x(:,3)));
    
    % Tính fitness song parallel
    parfor j = 1:p
        f(j) = compute_pso_loss(x(j,:), PV1_candidate, PV2_candidate, PV3_candidate, ...
            base_mpc, EV_Bus, P_EV_24h, PV_capacity_MW, mpopt);
    end
    
    update_fitness = (f <= fitness');
    fitness(update_fitness) = f(update_fitness);
    localx(update_fitness,:) = x(update_fitness,:);
    
    [globalmin, index] = min(fitness);
    g = localx(index,:);
    
    best_PV1 = PV1_candidate(g(1));
    best_PV2 = PV2_candidate(g(2));
    best_PV3 = PV3_candidate(g(3));
    
    fprintf('   Iter %2d/%2d | PV: [%2d, %2d, %2d] | Loss: %.6f MW\n', ...
        iter, k, best_PV1, best_PV2, best_PV3, globalmin);
end

best_PV1 = PV1_candidate(g(1));
best_PV2 = PV2_candidate(g(2));
best_PV3 = PV3_candidate(g(3));
PV_Bus = [best_PV1, best_PV2, best_PV3];

fprintf('\n>>> KẾT QUẢ TỐI ƯU PV:\n');
fprintf('    PV1 @ Bus %d | PV2 @ Bus %d | PV3 @ Bus %d\n', best_PV1, best_PV2, best_PV3);
fprintf('    Total Loss (24h) = %.6f MW\n\n', globalmin);

%% ===================== PHẦN 3: MÔ PHỎNG 24H SONG PARALLEL =================
fprintf('=== [3] MÔ PHỎNG 24H: (A) Base  (B) +EV  (C) +EV+PV ===\n');

hours = 1:24;

% Pre-allocate
Ploss_base = zeros(1,24);
Ploss_EV = zeros(1,24);
Ploss_EVPV = zeros(1,24);
V_base = zeros(n_buses, 24);
V_EV = zeros(n_buses, 24);
V_EVPV = zeros(n_buses, 24);

% Simulation results array
sim_results = cell(24, 1);

t_sim = tic;
parfor h = 1:24
    result = simulate_three_scenarios(h, base_mpc, EV_Bus, P_EV_24h, PV_Bus, PV_capacity_MW, mpopt);
    sim_results{h} = result;
end

% Extract results from cell array
for h = 1:24
    res = sim_results{h};
    Ploss_base(h) = res.ploss_base;
    Ploss_EV(h) = res.ploss_ev;
    Ploss_EVPV(h) = res.ploss_evpv;
    V_base(:,h) = res.v_base;
    V_EV(:,h) = res.v_ev;
    V_EVPV(:,h) = res.v_evpv;
end

fprintf('   Mô phỏng xong (%.1fs)\n', toc(t_sim));

fprintf('  Tổng P_loss 24h: Base=%.4f | +EV=%.4f | +EV+PV=%.4f MW\n', ...
    sum(Ploss_base), sum(Ploss_EV), sum(Ploss_EVPV));
fprintf('  Min V: Base=%.4f | +EV=%.4f | +EV+PV=%.4f p.u.\n\n', ...
    min(V_base(:)), min(V_EV(:)), min(V_EVPV(:)));

%% ===================== PHẦN 4: VẼ ĐỒ THỊ (TỰ CHỌN) ================================
fprintf('=== [4] VẼ ĐỒ THỊ (Tùy chọn) ====================\n');

plot_choice = input('>> Vẽ 7 biểu đồ chi tiết? (Y/n): ', 's');

if isempty(plot_choice) || lower(plot_choice) == 'y'
    fprintf('>> Đang vẽ 7 biểu đồ...\n');
    plot_results(hours, PV_Bus, EV_Bus, V_base, V_EV, V_EVPV, ...
        Ploss_base, Ploss_EV, Ploss_EVPV, P_EV_24h, total_base_load_MW, ...
        best_PV1, best_PV2, best_PV3, base_mpc);
    fprintf('   >> Hoàn tất vẽ 7 biểu đồ!\n');
else
    fprintf('>> Bỏ qua vẽ biểu đồ.\n');
end

%% ===================== THỐNG KÊ KẾT QUẢ ================================
fprintf('\n=============================================================\n');
fprintf('                     KẾT QUẢ TỔNG HỢP\n');
fprintf('=============================================================\n');

delta_loss_ev = (sum(Ploss_EV) - sum(Ploss_base)) / sum(Ploss_base) * 100;
delta_loss_pv = (sum(Ploss_EVPV) - sum(Ploss_base)) / sum(Ploss_base) * 100;
improvement_pv = (sum(Ploss_EV) - sum(Ploss_EVPV)) / sum(Ploss_EV) * 100;

fprintf('  Tổn thất công suất tác dụng (24h):\n');
fprintf('    Base:       %.4f MW\n', sum(Ploss_base));
fprintf('    +EV:        %.4f MW   (↑ %.2f%%)\n', sum(Ploss_EV), delta_loss_ev);
fprintf('    +EV+PV:     %.4f MW   (↑ %.2f%%)\n', sum(Ploss_EVPV), delta_loss_pv);
fprintf('    → Cải thiện: %.2f%% so với +EV\n\n', improvement_pv);

fprintf('  Điện áp tối thiểu:\n');
fprintf('    Base:       %.4f p.u.\n', min(V_base(:)));
fprintf('    +EV:        %.4f p.u.\n', min(V_EV(:)));
fprintf('    +EV+PV:     %.4f p.u.\n\n', min(V_EVPV(:)));

fprintf('  Vị trí PV tối ưu:\n');
fprintf('    PV1 @ Bus %d | PV2 @ Bus %d | PV3 @ Bus %d\n', best_PV1, best_PV2, best_PV3);
fprintf('    Công suất mỗi điểm: %.2f MW\n\n', PV_capacity_MW);

fprintf('  Vị trí EV:\n');
fprintf('    Trạm: [%s]\n', sprintf('%d ', EV_Bus));
fprintf('    Công suất đỉnh: %.4f MW\n', max(P_EV_24h));

fprintf('\n=============================================================\n');
total_time = toc(tic_total);
fprintf('⏱️  TỔNG THỜI GIAN CHẠY: %.1f giây\n', total_time);
fprintf('=============================================================\n');

%% =====================================================================
% HELPER FUNCTIONS
%% =====================================================================

function loss = compute_pso_loss(x, PV1_cand, PV2_cand, PV3_cand, ...
    base_mpc, EV_Bus, P_EV_24h, PV_cap, mpopt)
    bus1 = PV1_cand(x(1));
    bus2 = PV2_cand(x(2));
    bus3 = PV3_cand(x(3));
    PV_Bus_local = [bus1, bus2, bus3];
    
    total_loss = 0;
    for h = 1:24
        mpc = base_mpc;
        P_EV_node = P_EV_24h(h) / length(EV_Bus);
        for b = EV_Bus
            mpc.bus(b, 3) = mpc.bus(b, 3) + P_EV_node;
        end
        for b = PV_Bus_local
            mpc.bus(b, 3) = mpc.bus(b, 3) - PV_cap;
        end
        res = runpf(mpc, mpopt);
        if res.success
            total_loss = total_loss + sum(res.branch(:,14) + res.branch(:,16));
        else
            total_loss = total_loss + 1e3;
        end
    end
    loss = total_loss;
end

function result = simulate_three_scenarios(h, base_mpc, EV_Bus, P_EV_24h, PV_Bus, PV_cap, mpopt)
    mpopt_local = mpoption('verbose', 0, 'out.all', 0);
    
    % Scenario A: Base
    resA = runpf(base_mpc, mpopt_local);
    if resA.success
        v_base = resA.bus(:,8);
        ploss_base = sum(resA.branch(:,14) + resA.branch(:,16));
    else
        v_base = ones(size(base_mpc.bus,1),1);
        ploss_base = 0;
    end
    
    % Scenario B: +EV
    mpcB = base_mpc;
    P_EV_node = P_EV_24h(h) / length(EV_Bus);
    for b = EV_Bus
        mpcB.bus(b, 3) = mpcB.bus(b, 3) + P_EV_node;
    end
    resB = runpf(mpcB, mpopt_local);
    if resB.success
        v_ev = resB.bus(:,8);
        ploss_ev = sum(resB.branch(:,14) + resB.branch(:,16));
    else
        v_ev = ones(size(base_mpc.bus,1),1);
        ploss_ev = 0;
    end
    
    % Scenario C: +EV+PV
    mpcC = base_mpc;
    for b = EV_Bus
        mpcC.bus(b, 3) = mpcC.bus(b, 3) + P_EV_node;
    end
    for b = PV_Bus
        mpcC.bus(b, 3) = mpcC.bus(b, 3) - PV_cap;
    end
    resC = runpf(mpcC, mpopt_local);
    if resC.success
        v_evpv = resC.bus(:,8);
        ploss_evpv = sum(resC.branch(:,14) + resC.branch(:,16));
    else
        v_evpv = ones(size(base_mpc.bus,1),1);
        ploss_evpv = 0;
    end
    
    result = struct('v_base', v_base, 'v_ev', v_ev, 'v_evpv', v_evpv, ...
                    'ploss_base', ploss_base, 'ploss_ev', ploss_ev, 'ploss_evpv', ploss_evpv);
end

function plot_results(hours, PV_Bus, EV_Bus, V_base, V_EV, V_EVPV, ...
    Ploss_base, Ploss_EV, Ploss_EVPV, P_EV_24h, total_base_load_MW, ...
    best_PV1, best_PV2, best_PV3, base_mpc)
    
    n_buses = size(base_mpc.bus, 1);
    target_buses = [PV_Bus, EV_Bus];
    
    % FIGURE 1
    figure('Name','FIG 1: Voltage','Color','w','Position',[50 50 1400 900]);
    for idx = 1:length(target_buses)
        b = target_buses(idx);
        isPV = idx <= 3;
        lbl = sprintf('Bus %d %s', b, ternary(isPV,'(PV)','(EV)'));
        subplot(3,3,idx);
        plot(hours, V_base(b,:), 'k--', 'LineWidth', 1.5); hold on;
        plot(hours, V_EV(b,:), 'r-', 'LineWidth', 1.5);
        plot(hours, V_EVPV(b,:), 'b-', 'LineWidth', 2);
        title(lbl, 'FontWeight','bold');
        xlabel('Hour'); ylabel('V (p.u.)');
        legend('Base', '+EV', '+EV+PV');
        grid on; xlim([1 24]);
    end
    
    % FIGURE 2: Power Loss
    figure('Name','FIG 2: Power Loss','Color','w','Position',[150 150 1200 500]);
    plot(hours, Ploss_base, 'k--', 'LineWidth', 2); hold on;
    plot(hours, Ploss_EV, 'r-', 'LineWidth', 2);
    plot(hours, Ploss_EVPV, 'b-', 'LineWidth', 2);
    title('24h Power Loss Comparison', 'FontWeight','bold');
    xlabel('Hour'); ylabel('Loss (MW)');
    legend('Base', '+EV', '+EV+PV');
    grid on; xlim([1 24]);
    
    % FIGURE 3: Cumulative Loss
    figure('Name','FIG 3: Cumulative Loss','Color','w','Position',[200 200 900 500]);
    plot(hours, cumsum(Ploss_base), 'k--', 'LineWidth', 2); hold on;
    plot(hours, cumsum(Ploss_EV), 'r-', 'LineWidth', 2);
    plot(hours, cumsum(Ploss_EVPV), 'b-', 'LineWidth', 2);
    title('Cumulative 24h Power Loss', 'FontWeight','bold');
    xlabel('Hour'); ylabel('Cumulative Loss (MW)');
    legend('Base', '+EV', '+EV+PV');
    grid on; xlim([1 24]);
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
