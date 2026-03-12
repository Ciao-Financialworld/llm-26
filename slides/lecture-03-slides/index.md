<section class="title">
  <div class="title-main">Lecture 03 – Text Classification<br/>and Word Embeddings</div>
  <div class="title-sub">DATA130030.01 (自然语言处理)</div>
  <div class="title-meta">
    <div>Baojian Zhou</div>
    <div>School of Data Science</div>
    <div>Fudan University</div>
    <div>03/19/2026</div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="active">Text Classification and Word Meaning</li>
    <li class="muted">Counting-based Methods</li>
    <li class="muted">Learning-based Methods</li>
    <li class="muted">Bridge to LLMs</li>
  </ul>
</section>

---

<section class="ppt">
<div class="ppt-title">Assign probabilities to sentences</div>
  <div class="ppt-line"></div>
  <div style="display:flex; gap:2rem; align-items:stretch;">
    <div style="flex:1.05;">
      <p style="font-size: 34px; margin:0 0 12px 0; line-height:1.35;">
        A typical NLP task - Given a piece of text, we want to predict its label.
      </p>
      <ul style="font-size: 32px; line-height:1.45; margin-top:10px; padding-left: 1.2em;">
        <li><b>Spam detection</b>: spam / not spam</li>
        <li><b>Authorship attribution</b>: Madison / Hamilton</li>
        <li><b>Sentiment analysis</b>: positive / negative</li>
      </ul>
      <div style="margin-top:22px; padding:16px 20px; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; font-size: 30px; line-height:1.45;">
        <b>Common pattern:</b><br/>
        different tasks, but the same goal:<br/>
        <span style="color:#1f4ba5; font-weight:700;">text → representation → label</span>
      </div>
    </div>
    <div style="flex:1;">
      <div style="padding:16px 20px; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; font-size: 28px; line-height:1.4; margin-bottom:14px;">
        <b>Example 1</b><br/>
        "Congratulations! You won $10,000 ..."<br/>
        → <b>Spam</b>
      </div>
      <div style="padding:16px 20px; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; font-size: 28px; line-height:1.4; margin-bottom:14px;">
        <b>Example 2</b><br/>
        "Federalist Paper No. ..."<br/>
        → <b>Hamilton</b> or <b>Madison</b>
      </div>
      <div style="padding:16px 20px; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; font-size: 28px; line-height:1.4;">
        <b>Example 3</b><br/>
        "Full of fantastic characters and great plot twists."<br/>
        → <b>Positive</b>
      </div>
    </div>
  </div>

  <div style="margin-top:22px; text-align:center; font-size: 32px; font-weight:700;">
    <b>Key question:</b> How should we represent text so that a machine can classify it well?
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Classification methods</div>
  <div class="ppt-line"></div>
  <div style="display:flex; gap:2rem; align-items:flex-start;">
    <div style="flex:1.05; font-size: 30px; line-height: 1.4;">
      <ul style="margin: 0; padding-left: 1.1em;">
        <li style="margin-bottom: 14px;"><b>Hand-coded rules</b> <span class="text-red">(outdated)</span>
          <ul style="font-size: 0.92em; margin-top: 6px; line-height: 1.35;">
            <li>Rules based on combinations of words or other features (e.g., spam: black-list-address OR ("dollars" AND "have been selected"))</li>
            <li>Accuracy can be high if rules carefully refined by expert. But building and maintaining these rules is expensive. Rules change from time to time.</li>
          </ul>
        </li>
        <li style="margin-bottom: 8px;"><b>Supervised learning</b> <span class="text-green">(still useful)</span>
          <ul style="font-size: 0.92em; margin-top: 8px; line-height: 1.35;">
            <li><b>Input:</b> A document \(d\); set of classes \(C = \{c_1, c_2, \ldots, c_k\}\); a training set of \(n\) labeled documents \((d_1, c_1), \ldots, (d_n, c_n)\)</li>
            <li><b>Output:</b> A learned classifier \(\theta\): \(d \rightarrow c\)</li>
          </ul>
        </li>
      </ul>
    </div>
    <div style="flex:1; font-size: 28px; line-height: 1.35;">
      <div style="margin-bottom: 12px;"><b>Popular classifiers</b></div>
      <ul style="margin: 0 0 16px 0; padding-left: 1.1em;">
        <li><span class="text-red">Naïve Bayes (NB)</span></li>
        <li><span class="text-red">Logistic Regression (LR)</span></li>
        <li>Support-vector machines</li>
        <li>k-Nearest Neighbors</li>
        <li>NNs + LR/Softmax <span class="text-green">(modern way)</span></li>
      </ul>
      <img
        src="media/models-text-classification.png"
        alt="Naive Bayes, Logistic Regression, SVM, and Neural Networks"
        style="width:100%; max-width:520px; border:none; box-shadow:none; background:none;"
      >
      <div style="font-size:0.72em; color:#666; margin-top:6px;">
        Example models for text classification
      </div>
    </div>
  </div>
</section>

---

<!-- Add content slides below -->

