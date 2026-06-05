%% ========================================================================
% KẾT HỢP: 6 TRẠM EV (3 HÀNH VI) + TỐI ƯU 3 PV BẰNG PSO
% IEEE 33-bus | MATPOWER | 24h Power Flow
% Vẽ: Vt, It, DeltaP, DeltaQ tại các bus PV tìm được & bus EV
% 
% ⚡ PHIÊN BẢN TỐI ƯU HÓA: 100-200x nhanh hơn
% - Giảm PSO: 1000→50 particles, 100→30 iterations
% - Parallel: PSO, 24h simulation, fitness evaluation
% - Bỏ in log chi tiết, thêm tùy chọn vẽ biểu đồ
%% ========================================================================
clc; clear; close all;
tic; % Bắt đầu đếm thời gian

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
all_ts = cell(1,3); all_soc = cell(1,3); all_ds = cell(1,3);

for bv = 1:3
    rng(bv * 42);
    t_start = mod(normrnd(behaviors(bv).mu_t, behaviors(bv).sigma_t, [N_ev,1]), 24);
    t_start(t_start < 0) = t_start(t_start < 0) + 24;
    all_ts{bv} = t_start;
    
    soc_init = betarnd(behaviors(bv).alpha_soc, behaviors(bv).beta_soc, [N_ev,1]);
    all_soc{bv} = soc_init;
    
    distance = lognrnd(behaviors(bv).mu_d, behaviors(bv).sigma_d, [N_ev,1]);
    distance = min(distance, 500);
    all_ds{bv} = distance;
    
    P_h = zeros(1, 24);
    for i = 1:N_ev
        E_req = min(distance(i)*eff_distance, Cap_max_ev*(1-soc_init(i)));
        T_dur = E_req / (P_charge * eta_ev);
        hs = max(1, round(t_start(i)));
        he = min(24, hs + max(0, round(T_dur)));
        for h = hs:he
            P_h(h) = P_h(h) + P_charge;
        end
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

%% ===================== PHẦN 2: PSO TỐI ƯU VỊ TRÍ 3 PV (TỐI ƯU) ==================
fprintf('=== [2] PSO TỐI ƯU VỊ TRÍ 3 PV (TỔNG LOSS 24H) ===\n');

PV1_candidate = 2:33;
PV2_candidate = 2:33;
PV3_candidate = 2:33;
PV_capacity_MW = 0.4;

% ⚡ TỐI ƯU 1: Giảm kích thước swarm & iteration
n = 3;
p = 50;          % ⬇️ Giảm từ 1000 xuống 50 (20x)
k = 30;          % ⬇️ Giảm từ 100 xuống 30 (3.3x)
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

% ⚡ TỐI ƯU 2: Khởi tạo fitness song parallel & BỎ in log chi tiết
fprintf('>> Khởi tạo fitness cho %d particles (Parallel)...\n', p);
t_init = tic;
parfor i = 1:p
    f(i) = objective_PSO_PV_with_EV(x(i,:), PV1_candidate, PV2_candidate, PV3_candidate, ...
        base_mpc, EV_Bus, P_EV_24h, PV_capacity_MW, mpopt);
end
fitness = f;

[globalmin, index] = min(fitness);
g = x(index,:);
localx = x;

fprintf('>> Khởi tạo xong (%.1fs) | Best fitness = %.6f MW\n\n', toc(t_init), globalmin);

% Vòng lặp PSO (cải tiến)
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
    
    % ⚡ TỐI ƯU 3: Tính fitness song parallel
    parfor j = 1:p
        f(j) = objective_PSO_PV_with_EV(x(j,:), PV1_candidate, PV2_candidate, PV3_candidate, ...
            base_mpc, EV_Bus, P_EV_24h, PV_capacity_MW, mpopt);
        
        if f(j) <= fitness(j)
            fitness(j) = f(j);
            localx(j,:) = x(j,:);
        end
    end
    
    [globalmin, index] = min(fitness);
    g = localx(index,:);
    
    best_PV1 = PV1_candidate(g(1));
    best_PV2 = PV2_candidate(g(2));
    best_PV3 = PV3_candidate(g(3));
    
    % ⚡ TỐI ƯU 4: In log hiệu quả hơn (1 dòng mỗi vòng)
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

V_base = zeros(n_buses, 24); V_EV = zeros(n_buses, 24); V_EVPV = zeros(n_buses, 24);
Ploss_base = zeros(1,24); Ploss_EV = zeros(1,24); Ploss_EVPV = zeros(1,24);
Qloss_base = zeros(1,24); Qloss_EV = zeros(1,24); Qloss_EVPV = zeros(1,24);
I_branch_base = zeros(n_lines,24); I_branch_EV = zeros(n_lines,24); I_branch_EVPV = zeros(n_lines,24);
P_branch_loss_base = zeros(n_lines,24); P_branch_loss_EV = zeros(n_lines,24); P_branch_loss_EVPV = zeros(n_lines,24);
Q_branch_loss_base = zeros(n_lines,24); Q_branch_loss_EV = zeros(n_lines,24); Q_branch_loss_EVPV = zeros(n_lines,24);

% ⚡ TỐI ƯU: Chạy 24 giờ song parallel
t_sim = tic;
parfor h = 1:24
    % --- Scenario A: Chỉ tải nền ---
    mpcA = base_mpc;
    resA = runpf(mpcA, mpopt);
    if resA.success
        V_base(:,h) = resA.bus(:,8);
        Ploss_base(h) = sum(resA.branch(:,14) + resA.branch(:,16));
        Qloss_base(h) = sum(resA.branch(:,15) + resA.branch(:,17));
        S_brA = sqrt(resA.branch(:,14).^2 + resA.branch(:,15).^2);
        I_branch_base(:,h) = S_brA;
        P_branch_loss_base(:,h) = resA.branch(:,14) + resA.branch(:,16);
        Q_branch_loss_base(:,h) = resA.branch(:,15) + resA.branch(:,17);
    end
    
    % --- Scenario B: Tải nền + EV ---
    mpcB = base_mpc;
    P_EV_node = P_EV_24h(h) / length(EV_Bus);
    for b = EV_Bus
        mpcB.bus(b, 3) = mpcB.bus(b, 3) + P_EV_node;
    end
    resB = runpf(mpcB, mpopt);
    if resB.success
        V_EV(:,h) = resB.bus(:,8);
        Ploss_EV(h) = sum(resB.branch(:,14) + resB.branch(:,16));
        Qloss_EV(h) = sum(resB.branch(:,15) + resB.branch(:,17));
        S_brB = sqrt(resB.branch(:,14).^2 + resB.branch(:,15).^2);
        I_branch_EV(:,h) = S_brB;
        P_branch_loss_EV(:,h) = resB.branch(:,14) + resB.branch(:,16);
        Q_branch_loss_EV(:,h) = resB.branch(:,15) + resB.branch(:,17);
    end
    
    % --- Scenario C: Tải nền + EV + PV tối ưu ---
    mpcC = base_mpc;
    for b = EV_Bus
        mpcC.bus(b, 3) = mpcC.bus(b, 3) + P_EV_node;
    end
    for b = PV_Bus
        mpcC.bus(b, 3) = mpcC.bus(b, 3) - PV_capacity_MW;
    end
    resC = runpf(mpcC, mpopt);
    if resC.success
        V_EVPV(:,h) = resC.bus(:,8);
        Ploss_EVPV(h) = sum(resC.branch(:,14) + resC.branch(:,16));
        Qloss_EVPV(h) = sum(resC.branch(:,15) + resC.branch(:,17));
        S_brC = sqrt(resC.branch(:,14).^2 + resC.branch(:,15).^2);
        I_branch_EVPV(:,h) = S_brC;
        P_branch_loss_EVPV(:,h) = resC.branch(:,14) + resC.branch(:,16);
        Q_branch_loss_EVPV(:,h) = resC.branch(:,15) + resC.branch(:,17);
    end
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
    
    target_buses = [PV_Bus, EV_Bus];
    branch_to_target = zeros(1, length(target_buses));
    for i = 1:length(target_buses)
        b = target_buses(i);
        idx = find(base_mpc.branch(:,2) == b, 1);
        if ~isempty(idx)
            branch_to_target(i) = idx;
        else
            branch_to_target(i) = 0;
        end
    end
    
    colors = lines(9);
    
    %% --- Hàm helper thay thế sgtitle để tránh lỗi font ---
    function add_suptitle(str)
        axes('Position',[0 0 1 1],'Visible','off');
        text(0.5, 0.98, str, 'HorizontalAlignment','center', 'FontSize', 13, ...
            'FontWeight','bold', 'Units','normalized');
    end
    
    %% FIGURE 1: Vt TẠI CÁC NÚT PV & EV (3 SCENARIO)
    figure('Name','FIG 1: Vt tai cac nut PV & EV','Color','w','Position',[50 50 1400 900]);
    
    for idx = 1:length(target_buses)
        b = target_buses(idx);
        isPV = idx <= 3;
        lbl = sprintf('Bus %d %s', b, ternary(isPV,'(PV)','(EV)'));
        
        subplot(3,3,idx);
        plot(hours, V_base(b,:), 'k--', 'LineWidth', 1.5); hold on;
        plot(hours, V_EV(b,:), 'r-', 'LineWidth', 1.5);
        plot(hours, V_EVPV(b,:), 'b-', 'LineWidth', 2);
        yline(0.95, '--g', 'V_{min}=0.95', 'LineWidth', 1);
        yline(1.05, '--g', 'V_{max}=1.05', 'LineWidth', 1);
        title(lbl, 'FontWeight','bold');
        xlabel('Gio'); ylabel('V (p.u.)');
        legend('Base', '+EV', '+EV+PV', 'Location','best');
        grid on; xlim([1 24]); ylim([0.88 1.06]);
    end
    add_suptitle(sprintf('DIEN AP TUC THOI V_t TAI CAC NUT PV & EV | PV @ [%d %d %d] | EV @ [%s]', ...
        best_PV1, best_PV2, best_PV3, sprintf('%d ', EV_Bus)));
    
    %% FIGURE 2: It DÒNG NHÁNH ĐẾN CÁC NÚT PV & EV
    figure('Name','FIG 2: It nhanh den cac nut PV & EV','Color','w','Position',[100 100 1400 900]);
    
    for idx = 1:length(target_buses)
        b = target_buses(idx);
        br = branch_to_target(idx);
        isPV = idx <= 3;
        lbl = sprintf('Bus %d %s', b, ternary(isPV,'(PV)','(EV)'));
        
        subplot(3,3,idx);
        if br > 0
            plot(hours, I_branch_base(br,:), 'k--', 'LineWidth', 1.5); hold on;
            plot(hours, I_branch_EV(br,:), 'r-', 'LineWidth', 1.5);
            plot(hours, I_branch_EVPV(br,:), 'b-', 'LineWidth', 2);
            ylabel('S (MVA)');
        else
            text(0.5, 0.5, 'Slack Bus / Khong co nhanh den', 'HorizontalAlignment','center');
            axis off;
        end
        title(lbl, 'FontWeight','bold');
        xlabel('Gio');
        legend('Base', '+EV', '+EV+PV', 'Location','best');
        grid on; xlim([1 24]);
    end
    add_suptitle('DONG DIEN NHANH I_t DEN CAC NUT PV & EV');
    
    %% FIGURE 3: DeltaP TỔN THẤT TÁC DỤNG
    figure('Name','FIG 3: DeltaP ton that tac dung','Color','w','Position',[150 150 1400 700]);
    
    subplot(2,1,1);
    plot(hours, Ploss_base, 'k--', 'LineWidth', 2); hold on;
    plot(hours, Ploss_EV, 'r-', 'LineWidth', 2);
    plot(hours, Ploss_EVPV, 'b-', 'LineWidth', 2);
    title('dP_{total} - Ton that cong suat tac dung toan mang', 'FontWeight','bold');
    xlabel('Gio'); ylabel('dP (MW)');
    legend('Base', '+EV', '+EV+PV', 'Location','best');
    grid on; xlim([1 24]);
    
    subplot(2,1,2);
    hold on;
    for idx = 1:length(target_buses)
        br = branch_to_target(idx);
        if br > 0
            plot(hours, P_branch_loss_EVPV(br,:), 'LineWidth', 1.5, 'Color', colors(idx,:), ...
                'DisplayName', sprintf('Nhanh den Bus %d', target_buses(idx)));
        end
    end
    title('dP_{branch} - Ton that tac dung tai cac nhanh cap PV & EV (+EV+PV)', 'FontWeight','bold');
    xlabel('Gio'); ylabel('dP (MW)');
    legend('show', 'Location','best');
    grid on; xlim([1 24]);
    
    add_suptitle('TON THAT CONG SUAT TAC DUNG dP');
    
    %% FIGURE 4: DeltaQ TỔN THẤT PHẢN KHÁNG
    figure('Name','FIG 4: DeltaQ ton that phan khang','Color','w','Position',[200 200 1400 700]);
    
    subplot(2,1,1);
    plot(hours, Qloss_base, 'k--', 'LineWidth', 2); hold on;
    plot(hours, Qloss_EV, 'r-', 'LineWidth', 2);
    plot(hours, Qloss_EVPV, 'b-', 'LineWidth', 2);
    title('dQ_{total} - Ton that cong suat phan khang toan mang', 'FontWeight','bold');
    xlabel('Gio'); ylabel('dQ (MVAr)');
    legend('Base', '+EV', '+EV+PV', 'Location','best');
    grid on; xlim([1 24]);
    
    subplot(2,1,2);
    hold on;
    for idx = 1:length(target_buses)
        br = branch_to_target(idx);
        if br > 0
            plot(hours, Q_branch_loss_EVPV(br,:), 'LineWidth', 1.5, 'Color', colors(idx,:), ...
                'DisplayName', sprintf('Nhanh den Bus %d', target_buses(idx)));
        end
    end
    title('dQ_{branch} - Ton that phan khang tai cac nhanh cap PV & EV (+EV+PV)', 'FontWeight','bold');
    xlabel('Gio'); ylabel('dQ (MVAr)');
    legend('show', 'Location','best');
    grid on; xlim([1 24]);
    
    add_suptitle('TON THAT CONG SUAT PHAN KHANG dQ');
    
    %% FIGURE 5: SO SÁNH V-PROFILE TOÀN MẠNG TẠI GIỜ ĐỈNH
    [~, peak_h] = max(P_EV_24h + total_base_load_MW * 0.6);
    figure('Name','FIG 5: So sanh V-profile toan mang','Color','w','Position',[250 250 1000 500]);
    
    plot(1:n_buses, V_base(:,peak_h), 'k--o', 'LineWidth', 1.5); hold on;
    plot(1:n_buses, V_EV(:,peak_h), 'r-s', 'LineWidth', 1.5);
    plot(1:n_buses, V_EVPV(:,peak_h), 'b-^', 'LineWidth', 2);
    for i = 1:3
        plot(PV_Bus(i), V_EVPV(PV_Bus(i),peak_h), 'gp', 'MarkerSize', 14, 'MarkerFaceColor','g');
    end
    for i = 1:6
        plot(EV_Bus(i), V_EV(EV_Bus(i),peak_h), 'ms', 'MarkerSize', 10, 'MarkerFaceColor','m');
    end
    yline(0.95, '--g', 'V_{min}=0.95', 'LineWidth', 1.5);
    yline(1.05, '--g', 'V_{max}=1.05', 'LineWidth', 1.5);
    title(sprintf('Voltage Profile tai gio dinh (Gio %d)', peak_h), 'FontWeight','bold');
    xlabel('Bus'); ylabel('V (p.u.)');
    legend('Base', '+EV', '+EV+PV', 'PV location', 'EV station', 'V limits', 'Location','best');
    grid on; xlim([1 n_buses]); ylim([0.85 1.06]);
    
    %% FIGURE 6: TỔN THẤT TÍCH LŨY 24H
    figure('Name','FIG 6: Ton that tich luy 24h','Color','w','Position',[300 300 900 400]);
    
    cumP_base = cumsum(Ploss_base);
    cumP_EV = cumsum(Ploss_EV);
    cumP_EVPV = cumsum(Ploss_EVPV);
    
    plot(hours, cumP_base, 'k--', 'LineWidth', 2); hold on;
    plot(hours, cumP_EV, 'r-', 'LineWidth', 2);
    plot(hours, cumP_EVPV, 'b-', 'LineWidth', 2);
    title('Ton that tich luy 24h (Cumulative P_{loss})', 'FontWeight','bold');
    xlabel('Gio'); ylabel('Sum dP (MW)');
    legend('Base', '+EV', '+EV+PV', 'Location','best');
    grid on; xlim([1 24]);
    
    add_suptitle('TONG HOP TAC DONG CUA EV VA PV TOI UU');
    
    %% ===================== FIGURE 7: SƠ ĐỒ LƯỚI RADIAL ===================
    fprintf('   >> Vẽ sơ đồ lưới Radial...\n');
    
    figure('Name','FIG 7: So do luoi 33 bus Radial','Color','w','Position',[400 50 1200 700]);
    
    % Tọa độ (x,y) chính xác theo bố cục hình tia
    bus_x = zeros(33,1); 
    bus_y = zeros(33,1);
    
    % Nhánh chính ngang: 1 -> 2 -> ... -> 18
    for i = 1:18
        bus_x(i) = (i-1)*1.2;
        bus_y(i) = 0;
    end
    
    % Nhánh dưới 1 (từ bus 2): 19-20-21-22
    bus_x(19) = bus_x(2);       bus_y(19) = -2.5;
    bus_x(20) = bus_x(2)+1.2;   bus_y(20) = -2.5;
    bus_x(21) = bus_x(2)+2.4;   bus_y(21) = -2.5;
    bus_x(22) = bus_x(2)+3.6;   bus_y(22) = -2.5;
    
    % Nhánh dưới 2 (từ bus 3): 23-24-25
    bus_x(23) = bus_x(3);       bus_y(23) = -1.3;
    bus_x(24) = bus_x(3)+1.2;   bus_y(24) = -1.3;
    bus_x(25) = bus_x(3)+2.4;   bus_y(25) = -1.3;
    
    % Nhánh trên (từ bus 6): 26-27-28-29-30-31-32-33
    bus_x(26) = bus_x(6);       bus_y(26) = 2.5;
    bus_x(27) = bus_x(6)+1.2;   bus_y(27) = 2.5;
    bus_x(28) = bus_x(6)+2.4;   bus_y(28) = 2.5;
    bus_x(29) = bus_x(6)+3.6;   bus_y(29) = 2.5;
    bus_x(30) = bus_x(6)+4.8;   bus_y(30) = 2.5;
    bus_x(31) = bus_x(6)+6.0;   bus_y(31) = 2.5;
    bus_x(32) = bus_x(6)+7.2;   bus_y(32) = 2.5;
    bus_x(33) = bus_x(6)+8.4;   bus_y(33) = 2.5;
    
    hold on;
    
    % Vẽ các nhánh đường dây
    for k = 1:size(base_mpc.branch,1)
        fb = base_mpc.branch(k,1);
        tb = base_mpc.branch(k,2);
        plot([bus_x(fb), bus_x(tb)], [bus_y(fb), bus_y(tb)], ...
            'Color',[0.2 0.5 0.8], 'LineWidth', 2.5);
    end
    
    % Vẽ bus thường (tròn xanh nhạt)
    all_bus = 1:33;
    normal_bus = setdiff(all_bus, [EV_Bus, PV_Bus, 1]);
    scatter(bus_x(normal_bus), bus_y(normal_bus), 140, ...
        'MarkerEdgeColor',[0.2 0.5 0.8], 'MarkerFaceColor',[0.6 0.85 1], ...
        'LineWidth', 1.5);
    
    % Vẽ Slack Bus (1) - hình thoi vàng
    scatter(bus_x(1), bus_y(1), 220, 'd', ...
        'MarkerEdgeColor',[0.8 0.6 0], 'MarkerFaceColor',[1 0.9 0.3], ...
        'LineWidth', 2.5);
    
    % Vẽ 6 Trạm EV (tròn tím)
    scatter(bus_x(EV_Bus), bus_y(EV_Bus), 220, 'o', ...
        'MarkerEdgeColor',[0.5 0 0.6], 'MarkerFaceColor',[0.9 0.5 1], ...
        'LineWidth', 2);
    
    % Vẽ 3 Trạm PV tối ưu (ngôi sao vàng cam)
    scatter(bus_x(PV_Bus), bus_y(PV_Bus), 280, 'p', ...
        'MarkerEdgeColor',[0.9 0.5 0], 'MarkerFaceColor',[1 0.9 0.2], ...
        'LineWidth', 2.5);
    
    % Ghi nhãn số bus
    for i = 1:33
        if i == 1
            text(bus_x(i)-0.15, bus_y(i)+0.35, '1', 'FontSize', 10, ...
                'FontWeight','bold', 'Color',[0.5 0.4 0]);
        elseif ismember(i, EV_Bus)
            text(bus_x(i), bus_y(i)-0.45, num2str(i), 'HorizontalAlignment','center', ...
                'FontSize', 9, 'Color',[0.4 0 0.5], 'FontWeight','bold');
        elseif ismember(i, PV_Bus)
            text(bus_x(i), bus_y(i)+0.45, num2str(i), 'HorizontalAlignment','center', ...
                'FontSize', 10, 'Color',[0.8 0.4 0], 'FontWeight','bold');
        else
            text(bus_x(i), bus_y(i)-0.4, num2str(i), 'HorizontalAlignment','center', ...
                'FontSize', 9, 'Color',[0.2 0.3 0.5]);
        end
    end
    
    % Chú thích công suất bên cạnh PV
    for i = 1:3
        b = PV_Bus(i);
        text(bus_x(b)+0.35, bus_y(b)+0.25, sprintf('PV\\n%d:%.2fMW', b, PV_capacity_MW), ...
            'FontSize', 9, 'Color',[0.7 0.4 0], 'FontWeight','bold');
    end
    
    % Chú thích công suất EV
    P_EV_peak_node = max(P_EV_24h) / length(EV_Bus);
    for i = 1:length(EV_Bus)
        b = EV_Bus(i);
        text(bus_x(b)+0.35, bus_y(b)-0.35, sprintf('EV\\n%.2fMW', P_EV_peak_node), ...
            'FontSize', 8, 'Color',[0.5 0 0.6], 'FontWeight','bold');
    end
    
    % Legend
    h_slack = plot(NaN, NaN, 'd', 'MarkerSize', 12, 'MarkerEdgeColor',[0.8 0.6 0], ...
        'MarkerFaceColor',[1 0.9 0.3], 'LineWidth', 2);
    h_bus   = plot(NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor',[0.2 0.5 0.8], ...
        'MarkerFaceColor',[0.6 0.85 1], 'LineWidth', 1.5);
    h_ev    = plot(NaN, NaN, 'o', 'MarkerSize', 12, 'MarkerEdgeColor',[0.5 0 0.6], ...
        'MarkerFaceColor',[0.9 0.5 1], 'LineWidth', 2);
    h_pv    = plot(NaN, NaN, 'p', 'MarkerSize', 14, 'MarkerEdgeColor',[0.9 0.5 0], ...
        'MarkerFaceColor',[1 0.9 0.2], 'LineWidth', 2.5);
    h_line  = plot(NaN, NaN, '-', 'Color',[0.2 0.5 0.8], 'LineWidth', 2.5);
    
    legend([h_line, h_slack, h_bus, h_ev, h_pv], ...
        {'Duong day', 'Slack Bus (1)', 'Bus thuong', ...
        sprintf('EV station (6 tram) @ [%s]', sprintf('%d ', EV_Bus)), ...
        sprintf('PV toi uu (3 tram) @ [%d, %d, %d]', best_PV1, best_PV2, best_PV3)}, ...
        'Location','southeast', 'FontSize', 10, 'Box','on');
    
    title(sprintf('SO DO BO TRI PV TOI UU & EV TREN LUOI IEEE 33-BUS (Radial)\n'), ...
        'FontWeight','bold', 'FontSize', 13);
    
    axis equal;
    grid on;
    set(gca, 'XTick', [], 'YTick', [], 'Box', 'on');
    xlim([-1.5 22]); ylim([-4 4.5]);
    
    annotation('textbox', [0.15 0.01 0.7 0.06], 'String', ...
        sprintf('Ton that 24h: Base = %.3f MW  |  +EV = %.3f MW  |  +EV+PV toi uu = %.3f MW', ...
        sum(Ploss_base), sum(Ploss_EV), sum(Ploss_EVPV)), ...
        'EdgeColor','none', 'HorizontalAlignment','center', 'FontSize', 11, ...
        'BackgroundColor',[1 1 0.9], 'FontWeight','bold');
    
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
total_time = toc;
fprintf('⏱️  TỔNG THỜI GIAN CHẠY: %.1f giây\n', total_time);
fprintf('=============================================================\n');

%% =====================================================================
% LOCAL FUNCTIONS
%% =====================================================================

function loss = objective_PSO_PV_with_EV(x, PV1_cand, PV2_cand, PV3_cand, ...
    base_mpc, EV_Bus, P_EV_24h, PV_cap, mpopt)
    % Dùng chỉ số cột số trực tiếp để tránh lỗi "Unrecognized variable"
    % MATPOWER bus: PD=3, QD=4 | branch: PF=14, QF=15, PT=16, QT=17
    
    bus1 = PV1_cand(x(1));
    bus2 = PV2_cand(x(2));
    bus3 = PV3_cand(x(3));
    PV_Bus = [bus1, bus2, bus3];
    
    total_loss = 0;
    for h = 1:24
        mpc = base_mpc;
        % Thêm EV
        P_EV_node = P_EV_24h(h) / length(EV_Bus);
        for b = EV_Bus
            mpc.bus(b, 3) = mpc.bus(b, 3) + P_EV_node;
        end
        % Thêm PV (negative load)
        for b = PV_Bus
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

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
