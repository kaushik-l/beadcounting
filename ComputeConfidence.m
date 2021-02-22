%% max number of samples
n = 6;

%% generate possible outcomes
X = []; 
for k=1:n, X = [X ; dec2bin(sum(nchoosek(2.^(0:n-1),k),2)) - '0']; end
X = [zeros(1,n) ; X];

%% compute belief
q = 0.65; % probability of success
Xtot = cumsum(X,2);
Ntot = repmat(1:n, [2^n 1]);
b = 1./(1 + (q/(1-q)).^(Ntot - 2*Xtot));
b = [.5*ones(2^n,1) b]; % prepend a column of .5
p_nk = (q.^Xtot).*(1-q).^(Ntot-Xtot);
p_nk = [ones(2^n,1) p_nk].*repmat([1./2.^(n-(0:n))],[2^n 1]);
b(b<.5) = 1 - b(b<.5); 
b_exp = b.*p_nk;

%% plot
cmap = goodcolormap('bwr',2^n);
figure; hold on;
subplot(1,2,1); imagesc(0:n,1:2^n,b,[0 1]); colormap(cmap);
set(gca,'YDir','Normal');
set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',12); box off;
subplot(1,2,2); hold on;
for i=1:2^n, plot(0:n,b(i,:),'Linewidth',2,'Color',cmap(i,:)); end
set(gca,'TickDir','Out','Ticklength',[.03 .03],'Fontsize',12); box off;

figure; stem(0:n,sum(b_exp(1:64,:))); ylim([.5 1]);