document.addEventListener('DOMContentLoaded', () => {
    // DOM 元素选择器
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const consoleLog = document.getElementById('console-log');
    const clearLogBtn = document.getElementById('clear-log');

    const queryInput = document.getElementById('query-input');
    const searchBtn = document.getElementById('search-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = statusIndicator.querySelector('.status-text');
    const answerText = document.getElementById('answer-text');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const recalledList = document.getElementById('recalled-list');
    const rerankedList = document.getElementById('reranked-list');

    const detailModal = document.getElementById('detail-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const modalContent = document.getElementById('modal-chunk-content');
    const modalSource = document.getElementById('modal-chunk-source');
    const modalScore = document.getElementById('modal-chunk-score');

    // ==================== 检索模式切换 ====================
    let currentMode = 'langchain';  // 'langchain' | 'llamaindex'
    const modeOptions = document.querySelectorAll('.mode-option');
    const modelBadge = document.querySelector('.model-badge');

    function updateModeUI(mode) {
        modeOptions.forEach(opt => {
            opt.classList.toggle('active', opt.dataset.mode === mode);
        });
        currentMode = mode;

        if (mode === 'langchain') {
            modelBadge.innerHTML = `
                <span class="badge-item"><i class="fa-solid fa-cubes"></i> Qwen3-Reranker (0.6B)</span>
                <span class="badge-item"><i class="fa-solid fa-brain"></i> DeepSeek-V4-Flash</span>
            `;
            // Update diagnostic tab labels
            document.querySelector('[data-tab="tab-rerank"]').textContent = '精排重排 (Reranker)';
            // Update info alerts
            const rerankAlert = document.querySelector('#tab-rerank .intro-alert span');
            if (rerankAlert) rerankAlert.innerHTML = '以下是经过 <strong>qwen3-reranker-0.6b</strong> 交叉编码器精排排序后的文本，序号 1-3 将作为生成上下文喂给大模型。';
            const recallAlert = document.querySelector('#tab-recall .intro-alert span');
            if (recallAlert) recallAlert.innerHTML = '以下是直接从 <strong>Chroma 向量数据库</strong> 召回的最邻近候选文本块，尚未经过任何重排。';
        } else {
            modelBadge.innerHTML = `
                <span class="badge-item"><i class="fa-solid fa-route"></i> RouteQueryEngine</span>
                <span class="badge-item"><i class="fa-solid fa-brain"></i> DeepSeek-V4-Flash</span>
            `;
            document.querySelector('[data-tab="tab-rerank"]').textContent = '路由精排';
            const rerankAlert = document.querySelector('#tab-rerank .intro-alert span');
            if (rerankAlert) rerankAlert.innerHTML = 'LlamaIndex <strong>RouterQueryEngine</strong> 根据问题意图自动选择最合适的索引（语义/摘要/关键词），经 Qwen3-Reranker 精排后送大模型生成。';
            const recallAlert = document.querySelector('#tab-recall .intro-alert span');
            if (recallAlert) recallAlert.innerHTML = '由 <strong>RouterQueryEngine</strong> 选中的索引直接召回的结果，尚未经过 Reranker 精排。';
            // Hide route badge until next query
            const routeBadge = document.getElementById('route-badge');
            if (routeBadge) routeBadge.style.display = 'none';
        }
        log(`切换到 ${mode === 'langchain' ? 'LangChain' : 'LlamaIndex'} 检索模式`, 'system');
    }

    document.getElementById('mode-toggle').addEventListener('click', (e) => {
        const opt = e.target.closest('.mode-option');
        if (opt) updateModeUI(opt.dataset.mode);
    });

    // ==================== 自定义下拉选择框交互 ====================
    const customSelect = document.getElementById('custom-chunk-select');
    if (customSelect) {
        const trigger = customSelect.querySelector('.custom-select-trigger');
        const options = customSelect.querySelectorAll('.custom-option');
        const hiddenInput = document.getElementById('chunk-method');

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            customSelect.classList.toggle('open');
        });

        options.forEach(option => {
            option.addEventListener('click', (e) => {
                e.stopPropagation();
                const value = option.getAttribute('data-value');
                const text = option.textContent;

                // 更新触发器文本和值
                trigger.textContent = text;
                trigger.setAttribute('data-value', value);

                // 更新隐藏的 input 值供其它上传逻辑调用
                hiddenInput.value = value;

                // 更新选项高亮状态
                options.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');

                // 关闭菜单
                customSelect.classList.remove('open');
            });
        });

        // 点击外部关闭下拉菜单
        document.addEventListener('click', () => {
            customSelect.classList.remove('open');
        });
    }

    // ==================== 日志控制台功能 ====================
    function log(message, type = 'system') {
        const time = new Date().toLocaleTimeString();
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.innerHTML = `[${time}] ${message}`;
        consoleLog.appendChild(line);
        consoleLog.scrollTop = consoleLog.scrollHeight;
    }

    clearLogBtn.addEventListener('click', () => {
        consoleLog.innerHTML = `<div class="log-line system">[系统] 运行日志已清空。</div>`;
    });

    // ==================== 标签页切换功能 ====================
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // 切换按钮激活状态
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 切换内容区域显示
            tabContents.forEach(content => {
                if (content.id === targetTab) {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });
        });
    });

    // ==================== 文件拖拽上传 ====================
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove('dragover');
        }, false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    async function handleFileUpload(file) {
        const allowedExtensions = ['.txt', '.md', '.pdf', '.docx'];
        const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!allowedExtensions.includes(fileExt)) {
            log(`错误：文件 "${file.name}" 不是支持的格式，仅支持 .txt, .md, .pdf, .docx 文件。`, 'error');
            return;
        }
        
        const chunkMethod = document.getElementById('chunk-method').value;
        const chunkSize = parseInt(document.getElementById('chunk-size').value) || 500;
        const chunkOverlap = parseInt(document.getElementById('chunk-overlap').value) || 100;
        const methodText = chunkMethod === 'fixed' ? '固定长度切片' : '递归字符切片';
        
        log(`开始上传 "${file.name}" (${(file.size / 1024).toFixed(2)} KB)，分块模式：${methodText}(大小:${chunkSize}, 重叠:${chunkOverlap})...`, 'info');
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('chunk_method', chunkMethod);
        formData.append('chunk_size', chunkSize);
        formData.append('chunk_overlap', chunkOverlap);
        
        try {
            uploadZone.style.pointerEvents = 'none';
            uploadZone.style.opacity = '0.7';
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                if (result.added_count > 0) {
                    log(`上传成功！"${file.name}" 共切分为 ${result.chunks_count} 个文本块，并成功将 ${result.added_count} 个新文本块增量存入向量库。`, 'success');
                } else {
                    log(`上传完成："${file.name}" 共切片为 ${result.chunks_count} 个文本块，但它们都已存在于向量库中。未新增文本块。`, 'success');
                }
            } else {
                log(`错误：${result.message}`, 'error');
            }
        } catch (error) {
            log(`网络异常：无法上传文件 - ${error.message}`, 'error');
        } finally {
            uploadZone.style.pointerEvents = 'auto';
            uploadZone.style.opacity = '1';
            fileInput.value = ''; // 清空选择
        }
    }

    // ==================== RAG 查询问答 ====================
    async function performQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        // 设置 UI 为查询中状态
        searchBtn.disabled = true;
        queryInput.disabled = true;
        statusIndicator.classList.add('loading');
        statusText.innerText = '检索及生成中...';
        log(`[${currentMode === 'langchain' ? 'LangChain' : 'LlamaIndex'}] 发起检索提问: "${query}"`, 'info');

        // 清空前一次的回答
        answerText.innerHTML = `
            <div class="empty-answer">
                <i class="fa-solid fa-spinner fa-spin" style="color: var(--color-primary); font-size: 28px;"></i>
                <p>正在拉取候选数据并重排，请稍候...</p>
            </div>
        `;

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: query, mode: currentMode })
            });

            const result = await response.json();

            if (result.status === 'success') {
                const routeInfo = result.selected_route ? `[路由: ${result.selected_route}] ` : '';
                log(`${routeInfo}查询完成！召回了 ${result.recalled_chunks.length} 个候选文本，重排后为 DeepSeek 提供了 Top 3 上下文。`, 'success');

                // 显示路由选择（LlamaIndex 模式）
                const routeBadge = document.getElementById('route-badge');
                if (routeBadge && result.selected_route) {
                    routeBadge.style.display = 'inline-block';
                    routeBadge.className = `route-badge route-${result.selected_route}`;
                    const labels = { vector_tool: '语义检索', summary_tool: '摘要检索', keyword_tool: '关键词检索' };
                    const icons = { vector_tool: 'fa-solid fa-magnifying-glass', summary_tool: 'fa-solid fa-align-left', keyword_tool: 'fa-solid fa-key' };
                    routeBadge.innerHTML = `<i class="${icons[result.selected_route] || 'fa-solid fa-route'}"></i> ${labels[result.selected_route] || result.selected_route}`;
                } else if (routeBadge) {
                    routeBadge.style.display = 'none';
                }

                // 1. 渲染粗回（向量检索）结果
                renderRecalledList(result.recalled_chunks);

                // 2. 渲染精排（Rerank）结果
                renderRerankedList(result.reranked_chunks);

                // 3. 流式效果渲染 DeepSeek 回答
                typewriterEffect(result.answer);
            } else {
                log(`查询失败：${result.message}`, 'error');
                answerText.innerHTML = `<div style="color: var(--color-error);"><i class="fa-solid fa-triangle-exclamation"></i> 报错：${result.message}</div>`;
            }
        } catch (error) {
            log(`接口连接失败：${error.message}`, 'error');
            answerText.innerHTML = `<div style="color: var(--color-error);"><i class="fa-solid fa-triangle-exclamation"></i> 连接后端服务超时或遇到网络故障。</div>`;
        } finally {
            searchBtn.disabled = false;
            queryInput.disabled = false;
            statusIndicator.classList.remove('loading');
            statusText.innerText = '空闲';
        }
    }

    searchBtn.addEventListener('click', performQuery);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performQuery();
        }
    });

    // 渲染向量召回列表
    function renderRecalledList(chunks) {
        recalledList.innerHTML = '';
        if (chunks.length === 0) {
            recalledList.innerHTML = `<div class="empty-state"><i class="fa-solid fa-inbox"></i><p>向量数据库未返回任何匹配记录。</p></div>`;
            return;
        }

        chunks.forEach(chunk => {
            const card = document.createElement('div');
            card.className = 'chunk-card';
            card.innerHTML = `
                <div class="card-header-row">
                    <div class="card-meta-left">
                        <span class="card-badge rank">序号 #${chunk.index}</span>
                        <span class="card-badge source" title="${chunk.source}"><i class="fa-solid fa-file-invoice"></i> ${chunk.source}</span>
                    </div>
                </div>
                <div class="card-body-text">${escapeHtml(chunk.content)}</div>
            `;
            
            // 点击查看详情弹窗
            card.addEventListener('click', () => {
                showModal(chunk.content, chunk.source, `序号 #${chunk.index}`);
            });
            
            recalledList.appendChild(card);
        });
    }

    // 渲染重排列表
    function renderRerankedList(chunks) {
        rerankedList.innerHTML = '';
        if (chunks.length === 0) {
            rerankedList.innerHTML = `<div class="empty-state"><i class="fa-solid fa-inbox"></i><p>重排未返回结果。</p></div>`;
            return;
        }

        chunks.forEach(chunk => {
            const isSelected = chunk.rank <= 3;
            const card = document.createElement('div');
            card.className = `chunk-card ${isSelected ? 'selected-for-llm' : 'unused-for-llm'}`;
            card.innerHTML = `
                <div class="card-header-row">
                    <div class="card-meta-left">
                        <span class="card-badge rank">重排第 ${chunk.rank} 名</span>
                        <span class="card-badge source" title="${chunk.source}"><i class="fa-solid fa-file-invoice"></i> ${chunk.source}</span>
                    </div>
                    <div class="score-badge">相关度: ${typeof chunk.score === 'number' ? chunk.score.toFixed(4) : chunk.score}</div>
                </div>
                <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px;">
                    <i class="fa-solid fa-arrow-right-arrow-left"></i> 原始召回索引：#${chunk.original_index} 
                    ${isSelected ? ' <span style="color: var(--color-success); font-weight:600;">(已被喂给大模型)</span>' : ' (被重排算法淘汰)'}
                </div>
                <div class="card-body-text">${escapeHtml(chunk.content)}</div>
            `;
            
            // 点击查看详情弹窗
            card.addEventListener('click', () => {
                showModal(chunk.content, chunk.source, `重排第 ${chunk.rank} 名 / 关联评分: ${chunk.score}`);
            });
            
            rerankedList.appendChild(card);
        });
    }

    // 打字机流式输出效果
    function typewriterEffect(text) {
        answerText.innerHTML = '';
        let index = 0;
        const speed = 8; // 打字速度(毫秒/字)
        
        function type() {
            if (index < text.length) {
                // 判断是否包含换行等特殊渲染
                answerText.innerHTML = escapeHtml(text.slice(0, index + 1)) + '<span class="typing-cursor">█</span>';
                index++;
                setTimeout(type, speed);
            } else {
                answerText.innerHTML = escapeHtml(text); // 最终移除光标
            }
        }
        
        type();
    }

    // ==================== 模态弹窗控制 ====================
    function showModal(content, source, scoreInfo) {
        modalContent.textContent = content;
        modalSource.innerHTML = `<i class="fa-solid fa-file-invoice"></i> 来源：${source}`;
        modalScore.innerHTML = `<i class="fa-solid fa-calculator"></i> 归类：${scoreInfo}`;
        detailModal.classList.add('active');
    }

    function closeModal() {
        detailModal.classList.remove('active');
    }

    closeModalBtn.addEventListener('click', closeModal);
    detailModal.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            closeModal();
        }
    });

    // 辅助防注入转义函数
    function escapeHtml(string) {
        return String(string)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
