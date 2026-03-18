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

<section class="ppt">
  <div class="ppt-title">Naive Bayes (NB) Classifier</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:34px; align-items:flex-start; margin-top:18px;">
    <!-- Left column -->
    <div style="font-size:.7em; flex:1.05;">
      <ul class="bigul">
        <li>
          <b>Goal:</b> given a document \(d\), predict its most likely class \(c\)
        </li>
        <li class="fragment">
          <b>MAP decision rule:</b>
          \[
          c_{\text{MAP}}=\arg\max_{c\in C} P(c\mid d)
          \]
        </li>
        <li class="fragment">
          <b>By Bayes' rule: </b> posterior \(=\) likelihood \(\times\) prior / evidence
          \[
          P(c\mid d)=\frac{P(d\mid c)\,P(c)}{P(d)}
          \]
        </li>
        <li class="fragment">
          Since \(P(d)\) does not depend on \(c\),
          \[
          c_{\text{MAP}}
          = \arg\max_{c\in C} P(d\mid c)\,P(c)
          \]
        </li>
      </ul>
    </div>
    <!-- Right column -->
    <div style="flex:0.95;">
      <div style="background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:16px 18px; margin-bottom:16px;">
        <div style="font-size:.8em; font-weight:800; margin-bottom:10px; color:#1f4ba5;">
          Document representation
        </div>
        <div style="font-size:0.7em; line-height:1.45;">
          Represent document \(d\) as features
          \[
          d=(v_1,v_2,\dots,v_{|V|})
          \]
          using <b>bag-of-words</b>.
        </div>
      </div>
      <div class="fragment" style="background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:16px 18px;">
        <div style="font-size:.8em; font-weight:800; margin-bottom:10px; color:#1f4ba5;">
          NB classifier
        </div>
        <div style="font-size:0.7em; line-height:1.45;">
          \[
          c_{\text{MAP}}
          = \arg\max_{c\in C}
          P(d=(v_1,\dots,v_{|V|})\mid c)\,P(c)
          \]
        </div>
        <div style="margin-top:10px; font-size:0.7em; color:#444;">
          The remaining question is: how should we model \(P(d\mid c)\)?
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Naive Bayes: Bag-of-Words Model</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:34px; align-items:flex-start; margin-top:18px;">
    <!-- Left -->
    <div style="flex:1.02; font-size:0.8em">
      <ul class="bigul">
        <li>
          <b>Step 1: represent a document as features</b>
        </li>
      </ul>
      <div style="margin:10px 0 16px 22px; padding:14px 16px; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; font-size:0.8em; line-height:1.35;">
        <b>Example document</b><br>
        “I love this movie. It’s sweet, witty, and great.”
      </div>
      <div style="margin-left:22px; font-size:0.7em; line-height:1.4;">
        Keep only a vocabulary-based feature vector:
        \[
        d \;\longrightarrow\; x=(x_1,x_2,\dots,x_{|V|})
        \]
        where \(x_i\) is the count of word \(v_i\) in the document.
      </div>
      <div style="margin:14px 0 0 22px; padding:12px 14px; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; font-size:0.7em;">
        <b>Bag-of-words:</b> word order is ignored; only word counts matter.
      </div>
    </div>
    <!-- Right -->
    <div style="flex:1.0; font-size:0.7em">
      <ul class="bigul">
        <li class="fragment">
          <b>Step 2: classify by MAP</b>
          \[
          c_{\text{MAP}}=\arg\max_{c\in C} P(c\mid d)
          \]
          \[
          c_{\text{MAP}}=\arg\max_{c\in C} P(d\mid c)\,P(c)
          \]
        </li>
        <li class="fragment">
          <b>Naive Bayes assumption:</b> features are conditionally independent given class \(c\)
          \[
          P(d\mid c)\approx \prod_{i=1}^{|V|} P(v_i\mid c)^{x_i}
          \]
        </li>
      </ul>
      <b>Why “naive”?</b><br>
        In reality, words are correlated, but this approximation often works surprisingly well.
    </div>

  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Training and Prediction in Naive Bayes</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:34px; align-items:flex-start; margin-top:18px;">
    <!-- Left -->
    <div style="flex:1.02; font-size:0.8em">
      <ul class="bigul">
        <li>
          <b>Estimate the class prior</b>
          \[
          \hat P(c)=\frac{N_c}{n}
          \]
          <div style="font-size:0.7em; color:#555; margin-top:4px;">
            \(n\): total number of training documents, \(\;N_c\): number of documents in class \(c\)
          </div>
        </li>
        <li class="fragment">
          <b>Estimate word probabilities in each class</b>
          \[
          \hat P(v_i\mid c)=
          \frac{\mathrm{count}(v_i,c)+\alpha}
          {\sum_{v\in V}\mathrm{count}(v,c)+\alpha |V|}
          \]
        </li>
        <li class="fragment">
          <b>Smoothing is important</b><br>
          use \(\alpha=1\) (Laplace/add-one smoothing) to avoid zero probabilities
        </li>
      </ul>
    </div>
    <!-- Right -->
    <div style="flex:1.0;">
      <div style="padding:14px 16px; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; margin-bottom:14px;">
        <div style="font-size:.8em; font-weight:800; margin-bottom:8px; color:#1f4ba5;">
          Prediction rule
        </div>
        <div style="font-size:0.7em; line-height:1.45;">
          For a test document \(d\) with count vector \(x\),
          \[
          c_{\text{NB}}
          =
          \arg\max_{c\in C}
          \left[
          \log P(c)+
          \sum_{i=1}^{|V|} x_i \log P(v_i\mid c)
          \right]
          \]
        </div>
      </div>
      <div class="fragment" style="padding:14px 16px; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px;">
        <div style="font-size:.8em; font-weight:800; margin-bottom:8px; color:#1f4ba5;">
          Why take logs?
        </div>
        <ul class="subul" style="margin:0; font-size:0.7em; line-height:1.4;">
          <li>turn products into sums</li>
          <li>avoid numerical underflow</li>
          <li>make computation simpler and more stable</li>
        </ul>
      </div>
      <div class="fragment" style="margin-top:14px; text-align:center; font-size:0.9em; font-weight:700; color:#1f4ba5;">
        Train \(P(c)\) and \(P(v_i\mid c)\) from counts, then score each class.
      </div>
    </div>

  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Logistic Regression (LR) Classifier</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:34px; align-items:flex-start; margin-top:18px;">
    <!-- Left column -->
    <div style="flex:1.02; font-size:0.8em">
      <ul class="bigul">
        <li>
          <b>Input:</b> each document is represented by a feature vector
          \[
          \mathbf{x} = [x_1, x_2, \dots, x_p]^\top
          \]
        </li>
        <li class="fragment">
          <b>Idea:</b> each feature \(x_i\) has a weight \(w_i\)
          measuring how important it is for prediction
        </li>
      </ul>
      <div class="fragment" style="margin:10px 0 0 22px; padding:14px 16px; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; font-size:0.7em; line-height:1.45;">
        <b>Sentiment example</b><br><br>
        <span style="color:#3f6cc8;">awesome</span>  \(\rightarrow\) positive weight<br>
        <span style="color:#3f6cc8;">abysmal</span> \(\rightarrow\) negative weight<br>
        <span style="color:#3f6cc8;">mediocre</span> \(\rightarrow\) slightly negative weight
      </div>
      <div class="fragment" style="margin:14px 0 0 22px; padding:14px 16px; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; font-size:0.7em; line-height:1.4;">
        <b>Linear score $z = \mathbf{w}^\top \mathbf{x} + b$</b>
        where \(\mathbf{w}=[w_1,w_2,\dots,w_p]^\top\) and \(b\) is the bias.
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.0; font-size:0.7em">
      <ul class="bigul">
        <li class="fragment">
          We want a <b>probabilistic classifier</b>, not just a raw score
        </li>
        <li class="fragment">
          Convert the score into a probability using the <b>sigmoid</b>:
          \[
          P(y=1\mid \mathbf{x}) = \sigma(z)
          = \frac{1}{1+e^{-z}}
          \]
        </li>
        <li class="fragment">
          Therefore, $P(y=1\mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}+b)$
        </li>
        <li class="fragment">
          Predict class \(1\) if the probability is high enough,
          otherwise predict class \(0\)
        </li>
      </ul>
      <div class="fragment" style="margin-top:10px; padding:14px 16px; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; font-size:0.7em; line-height:1.4;">
        <b>Interpretation:</b><br>
        <li>
        - LR directly models \(P(y\mid \mathbf{x})\), unlike Naive Bayes,
        which models \(P(\mathbf{x}\mid y)\) and then applies Bayes’ rule.
        </li>
        <li>
        - feature vector \(\;\rightarrow\;\) linear score \(\;\rightarrow\;\) sigmoid \(\;\rightarrow\;\) probability
        </li>
      </div>
    </div>
  </div>

</section>

---

<section class="ppt">
  <div class="ppt-title">Logistic Regression: Binary and Multinomial</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Binary Logistic Regression
      </div>
      <div style="font-size:0.7em; line-height:1.4; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Compute a linear score $z = \mathbf{w}^{\top}\mathbf{x} + b$
        </div>
        <div style="margin-bottom:8px;">
          Turn it into a probability by the <b>sigmoid</b>:
          \[
          \sigma(z)=\frac{1}{1+e^{-z}}
          \]
        </div>
        <div style="margin-bottom:8px;">
          Therefore,
          \[
          P(y=1\mid \mathbf{x})=\sigma(\mathbf{w}^{\top}\mathbf{x}+b)
          \]
          \[
          P(y=0\mid \mathbf{x})=1-\sigma(\mathbf{w}^{\top}\mathbf{x}+b)=\sigma(-z)
          \]
        </div>
        <div>
          <b>Interpretation:</b> sigmoid maps an arbitrary score \(z\) into \([0,1]\),
          so the output can be interpreted as a probability.
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Multinomial Logistic Regression (Softmax)
      </div>
      <div style="font-size:0.6em; line-height:1.4; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          For \(k\) classes, compute one score for each class:
          \[
          z_c=\mathbf{w}_c^{\top}\mathbf{x}+b_c,\qquad c=1,\dots,k
          \]
        </div>
        <div style="margin-bottom:8px;">
          Convert the score vector \(\mathbf{z}=[z_1,\dots,z_k]\) into probabilities
          using <b>softmax</b>:
          \[
          \mathrm{softmax}(z_c)=\frac{\exp(z_c)}{\sum_{j=1}^{k}\exp(z_j)}
          \]
        </div>
        <div style="margin-bottom:8px;">
          Hence,
          \[
          P(y=c\mid \mathbf{x})
          =
          \frac{\exp(\mathbf{w}_c^{\top}\mathbf{x}+b_c)}
          {\sum_{j=1}^{k}\exp(\mathbf{w}_j^{\top}\mathbf{x}+b_j)}
          \]
        </div>
        <div>
          <b>Interpretation:</b> softmax is the multiclass extension of sigmoid,
          producing a valid probability distribution over all classes.
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:16px; text-align:center; font-size:0.8em; font-weight:700; color:#1f4ba5;">
    score \(\rightarrow\) probability:
    &nbsp; binary use <b>sigmoid</b>, multiclass use <b>softmax</b>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Logistic Regression: Loss Function</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Why do we need a loss?
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <ul style="margin:0 0 0 1.1em; padding:0;">
          <li>In supervised classification, each input \(\mathbf{x}\) has a true label \(y\in\{0,1\}\).</li>
          <li>The model outputs a probability estimate
            \[
            \hat{y}=\sigma(\mathbf{w}^{\top}\mathbf{x}+b).
            \]
          </li>
          <li>We need a function \(L(\hat{y},y)\) to measure how far the prediction is from the true label.</li>
          <li>Training means choosing \(\mathbf{w}\) and \(b\) to <b>minimize</b> this loss over the training set.</li>
        </ul>
      </div>
      <div style="margin-top:12px; font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Key idea
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        For logistic regression, the natural choice is the
        <b>negative log-likelihood</b>, which is also called the
        <b>cross-entropy loss</b>.
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Cross-entropy loss for binary LR
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Logistic regression models the probability of the true label as
          \[
          P(y\mid \mathbf{x})=\hat{y}^{\,y}(1-\hat{y})^{\,1-y},
          \qquad \hat{y}=\sigma(\mathbf{w}^{\top}\mathbf{x}+b).
          \]
        </div>
        <div style="margin-bottom:8px;">
          Taking logs gives the log-likelihood:
          \[
          \log P(y\mid \mathbf{x})
          = y\log \hat{y} + (1-y)\log(1-\hat{y}).
          \]
        </div>
        <div style="margin-bottom:8px;">
          To obtain something to <b>minimize</b>, flip the sign:
          \[
          L_{CE}(\hat{y},y)
          = -\Big[y\log \hat{y} + (1-y)\log(1-\hat{y})\Big].
          \]
        </div>
        <div>
          Substituting \(\hat{y}=\sigma(\mathbf{w}^{\top}\mathbf{x}+b)\):
          \[
          L_{CE}
          =-\Big[
          y\log \sigma(\mathbf{w}^{\top}\mathbf{x}+b)
          +(1-y)\log\big(1-\sigma(\mathbf{w}^{\top}\mathbf{x}+b)\big)
          \Big].
          \]
          Maximize likelihood \(\;\Longleftrightarrow\;\) Minimize cross-entropy
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Logistic Regression: Objective and GD</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Training objective
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          The model parameters are
          \[
          \theta=(\mathbf{w}, b).
          \]
        </div>
        <div style="margin-bottom:8px;">
          For each example \((\mathbf{x}^{(i)}, y^{(i)})\), logistic regression produces
          \[
          \hat y^{(i)} = f(\mathbf{x}^{(i)};\theta).
          \]
        </div>
        <div style="margin-bottom:8px;">
          We choose \(\theta\) to minimize the average loss:
          \[
          \hat\theta
          =
          \arg\min_{\theta}\;
          \ell(\theta)
          =
          \arg\min_{\theta}\;
          \frac{1}{n}\sum_{i=1}^{n}
          L\!\big(f(\mathbf{x}^{(i)};\theta),\, y^{(i)}\big).
          \]
        </div>
        <div>
          For logistic regression, \(\ell(\theta)\) is a <b>convex</b> function.
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Gradient descent
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          The gradient $\nabla \ell(\theta)$ points in the direction of the greatest increase of the loss. So to make the loss smaller, we move in the <b>opposite direction</b>.
        </div>
        <div style="margin-bottom:8px;">
          Gradient descent update:
          \[
          \theta^{t+1}
          =
          \theta^{t}
          -
          \eta \nabla \ell(\theta^{t}),
          \]
          where \(\eta\) is the learning rate.
        </div>
        <div>
          <b>Interpretation:</b> repeat “compute gradient \(\rightarrow\) move downhill”
          until the loss stops decreasing. objective \(\rightarrow\) gradient \(\rightarrow\) update
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Logistic Regression: One Gradient Descent Step</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Toy example setup
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Suppose the true label is
          \[
          y=1
          \]
          and the feature vector is
          \[
          \mathbf{x}=[x_1,x_2]^\top=[3,2]^\top.
          \]
        </div>
        <div style="margin-bottom:8px;">
          Initialize
          \[
          w_1=w_2=b=0,\qquad \eta=0.1.
          \]
        </div>
        <div>
          So initially,
          \[
          z=\mathbf{w}^\top \mathbf{x}+b=0,\qquad \sigma(z)=\sigma(0)=0.5.
          \]
        </div>
        One step of GD adjusts the weights to increase the probability of the correct label
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.7em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Compute gradient and update
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          For logistic regression,
          \[
          \frac{\partial L}{\partial w_j}
          =
          \big(\sigma(\mathbf{w}^\top\mathbf{x}+b)-y\big)x_j,
          \qquad
          \frac{\partial L}{\partial b}
          =
          \sigma(\mathbf{w}^\top\mathbf{x}+b)-y.
          \]
        </div>
        <div style="margin-bottom:8px;">
          Since \(\sigma(0)=0.5\) and \(y=1\), $\sigma(0)-y = -0.5$. Therefore,
        </div>
        <div style="margin-bottom:8px;">
          \[
          \nabla_\theta L
          =
          \begin{bmatrix}
          -0.5\cdot 3 \\
          -0.5\cdot 2 \\
          -0.5
          \end{bmatrix}
          =
          \begin{bmatrix}
          -1.5\\
          -1.0\\
          -0.5
          \end{bmatrix}.
          \]
        </div>
        <div>
          Update:
          \[
          \theta^{1}
          =
          \theta^{0}-\eta \nabla_\theta L
          =
          \begin{bmatrix}0\\0\\0\end{bmatrix}
          -
          0.1
          \begin{bmatrix}-1.5\\-1.0\\-0.5\end{bmatrix}
          =
          \begin{bmatrix}
          0.15\\
          0.10\\
          0.05
          \end{bmatrix}.
          \]
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Logistic Regression: Training in Practice</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Optimization objective
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Logistic regression learns parameters $\theta=(\mathbf{w},b)$ by minimizing the average cross-entropy loss:
          \[
          \hat\theta
          =
          \arg\min_{\theta}\;
          \frac{1}{n}\sum_{i=1}^{n}
          L\!\big(f(\mathbf{x}^{(i)};\theta),\, y^{(i)}\big).
          \]
        </div>
        <div>
          We update parameters by gradient descent: $\theta^{t+1} = \theta^{t} - \eta \nabla \ell(\theta^{t})$, where \(\eta\) is the learning rate.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Full-batch, SGD, and mini-batch
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          <b>Full-batch:</b> use the whole dataset for each update
        </div>
        <div style="margin-bottom:6px;">
          <b>SGD:</b> use one example at a time
        </div>
        <div>
          <b>Mini-batch:</b> use a small batch of examples; this is the modern default
          because it is efficient and stable
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Generalization and overfitting
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          In text classification, rare words or rare \(n\)-grams may look highly predictive
          on the training set but fail on new examples.
        </div>
        <div>
          This is called <b>overfitting</b>: high training accuracy,
          but poor test performance.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Common remedy
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Add regularization to penalize overly large weights:
          \[
          \ell_{\text{reg}}(\theta)
          =
          \ell(\theta) + \lambda \|\mathbf{w}\|_2^2.
          \]
        </div>
        <div>
          In practice: use a validation set, mini-batch optimization,
          and regularization to improve generalization.
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">From NB and LR to Neural Text Classification</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Two classic views
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:10px;">
          <b>Naive Bayes (NB): generative</b><br>
          model the joint distribution by $P(y,\mathbf{x}) = P(y)\,P(\mathbf{x}\mid y).$ Then classify by Bayes’ rule.
        </div>
        <div>
          <b>Logistic Regression (LR): discriminative</b><br>
          directly model $P(y\mid \mathbf{x})$. Learn weights that separate classes well.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Intuition
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          <b>NB:</b> explain how each class could generate the text
        </div>
        <div>
          <b>LR:</b> focus only on how to distinguish one class from another
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Bridge to neural classifiers
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Classic models usually rely on manually designed features
          such as bag-of-words or \(n\)-gram counts. Neural models replace hand-crafted features with a learned representation: $\textbf{Embeddings!}$
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Key takeaway
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          NB and LR are still important because they give the basic ideas:
          probabilistic modeling, decision rules, and linear classification.
        </div>
        <div>
          Neural text classifiers keep the same classification goal,
          but learn better representations from data.
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    classic models use fixed features; neural models learn the features
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="muted">Text Classification and Word Meaning</li>
    <li class="active">Counting-based Methods</li>
    <li class="muted">Learning-based Methods</li>
    <li class="muted">Bridge to LLMs</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">Word Meaning: From Symbols to Semantics</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Discrete symbols are limited
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          In \(n\)-gram models, each word is treated as a separate symbol.
        </div>
        <div style="margin-bottom:8px;">
          A simple representation is a <b>one-hot vector</b>:
          \[
          \text{hotel} \rightarrow \mathbf{e}_{\text{hotel}},\qquad
          \text{motel} \rightarrow \mathbf{e}_{\text{motel}}
          \]
          where the dimension is \(|V|\).
        </div>
        <div style="margin-bottom:8px;">
          But one-hot vectors do not tell us that
          <b>hotel</b> and <b>motel</b> are semantically close.
        </div>
        <div>
          So discrete symbols are useful for indexing words,
          but not for representing <b>meaning</b>.
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        What should a meaning representation capture?
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          <b>Synonymy:</b> car / automobile, couch / sofa
        </div>
        <div style="margin-bottom:6px;">
          <b>Antonymy:</b> hot / cold, rise / fall
        </div>
        <div style="margin-bottom:6px;">
          <b>Similarity:</b> cat / dog, car / bicycle
        </div>
        <div>
          <b>Relatedness:</b> doctor / hospital, banking / money
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Key idea: distributional hypothesis
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px; color:#c62828; font-weight:700;">
          Words that occur in similar contexts tend to have similar meanings.
        </div>
        <div style="font-style:italic;">
          “You shall know a word by the company it keeps.” — J. R. Firth
        </div>
      </div>
    </div>
  </div>

  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Word meaning should come from how a word is used in context
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Distributional Hypothesis: A Simple Example</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Suppose we do not know the word “ongchoi”
      </div>
      <div style="font-size:0.7em; line-height:1.5; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          We see the following contexts:
        </div>
        <div style="margin-bottom:6px;">
          S1: Ongchoi is <span style="color:#12a150;">delicious sautéed with garlic</span>.
        </div>
        <div style="margin-bottom:6px;">
          S2: Ongchoi is <span style="color:#12a150;">superb over rice</span>.
        </div>
        <div>
          S3: Ongchoi <span style="color:#12a150;">leaves with salty sauces</span>.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Inference from context
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        From words like <b>leaves</b>, <b>delicious</b>, and <b>sautéed</b>,
        we can infer that <b>ongchoi</b> is probably a leafy food or vegetable.
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        What does this example show?
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          We do not need a dictionary entry first.
        </div>
        <div style="margin-bottom:8px;">
          Meaning can be inferred from the <b>surrounding words</b>.
        </div>
        <div>
          So a word can be represented by the contexts in which it appears.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Bridge to the next topic
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          This motivates representing each word by
          <b>statistics of its context</b>.
        </div>
        <div>
          Next, we will see how to build such representations with
          <b>counting-based methods</b>.
        </div>
      </div>
    </div>
  </div>

  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Context gives us a practical way to represent meaning
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Word Embeddings: A Formal Definition</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Vocabulary and embedding map
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Let the vocabulary be
          \[
          V=\{w_1,w_2,\dots,w_{|V|}\}.
          \]
        </div>
        <div style="margin-bottom:8px;">
          A <b>word embedding</b> is a mapping $\phi: V \rightarrow \mathbb{R}^{d},$ which assigns each word \(w\) a vector
          \[
          \phi(w)=\mathbf{e}_w \in \mathbb{R}^d.
          \]
        </div>
        <div>
          Usually \(d \ll |V|\), so embeddings are a low-dimensional
          representation of words.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Embedding matrix
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        If we stack all vectors together, we get an embedding matrix $E \in \mathbb{R}^{|V|\times d},$ where the \(i\)-th row \(E_{i,:}\) is the embedding of word \(w_i\).
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        What should embeddings preserve?
      </div>
      <div style="font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Words with similar meanings should have similar vectors. Similarity is often measured by cosine similarity:
          \[
          \mathrm{sim}(w_i,w_j)
          =
          \frac{\mathbf{e}_{w_i}^{\top}\mathbf{e}_{w_j}}
          {\|\mathbf{e}_{w_i}\|\;\|\mathbf{e}_{w_j}\|}.
          \]
        </div>
        <div>
          So embeddings turn semantic similarity into
          <b>geometric closeness</b> in vector space.
        </div>
      </div>
      <br/>
      <div style="font-size:0.7em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        Intuition: Instead of representing a word as a one-hot symbol,
        we represent it by a vector whose position reflects how the word is used.
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">How Do We Obtain Word Embeddings?</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        1. Counting-based embeddings
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Build a word-context co-occurrence matrix
          \[
          X \in \mathbb{R}^{|V|\times |C|},
          \]
          where \(X_{ij}\) counts how often word \(w_i\) appears with context \(c_j\). Then define the embedding of \(w_i\) from the \(i\)-th row:
          \[
          \mathbf{e}_{w_i} = X_{i,:}
          \quad \text{or} \quad
          \mathbf{e}_{w_i}=M_{i,:},
          \]
          where \(M\) may be a transformed matrix such as TF-IDF or PPMI.
          These vectors are often <b>sparse</b>, and can optionally be reduced
          to a dense low-dimensional space by matrix factorization.
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        2. Learning-based embeddings
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Treat the embedding matrix \(E\) as trainable parameters. Learn \(E\) by optimizing an objective based on nearby words, e.g.
          \[
          \max_E \sum_{t}\sum_{j\in \mathcal{N}(t)}
          \log P(w_{t+j}\mid w_t).
          \]
        </div>
        <div style="margin-bottom:8px;">
          Models such as <b>word2vec</b>, <b>GloVe</b>, and <b>fastText</b>
          learn dense vectors so that words occurring in similar contexts get similar embeddings.
        </div>
        <div>
          These embeddings are usually <b>dense</b> and low-dimensional.
        </div>
      </div>
    </div>
  </div>

  <div style="margin-top:14px; display:flex; gap:18px;">
    <div style="flex:1; font-size:0.6em; line-height:1.38; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:10px 12px;">
      <b>Counting-based view:</b><br>
      a word is represented by observed context statistics
    </div>
    <div style="flex:1; font-size:0.6em; line-height:1.38; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:10px 12px;">
      <b>Learning-based view:</b><br>
      a word is represented by parameters learned from a prediction objective
    </div>
  </div>
  <div style="margin-top:12px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    two routes to embeddings: count contexts directly, or learn vectors by prediction
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Counting-based Representations</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Two common matrices
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          <b>Term-document matrix</b><br>
          rows = words, columns = documents
        </div>
        <div style="margin-bottom:8px;">
          <b>Word-context / co-occurrence matrix</b><br>
          rows = target words, columns = context words
        </div>
        <div>
          In both cases, each row gives a count-based vector representation.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Small example
      </div>
      <div style="font-size:0.6em; line-height:1.35; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:10px 12px;">
        <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.95em;">
          <tr style="background:#4d79c7; color:white;">
            <th style="padding:4px;"></th>
            <th style="padding:4px;">Doc 1</th>
            <th style="padding:4px;">Doc 2</th>
            <th style="padding:4px;">Doc 3</th>
          </tr>
          <tr><td style="padding:4px; font-weight:700;">battle</td><td>1</td><td>0</td><td>7</td></tr>
          <tr style="background:#eef2fb;"><td style="padding:4px; font-weight:700;">good</td><td>114</td><td>80</td><td>62</td></tr>
          <tr><td style="padding:4px; font-weight:700;">fool</td><td>36</td><td>58</td><td>1</td></tr>
          <tr style="background:#eef2fb;"><td style="padding:4px; font-weight:700;">wit</td><td>20</td><td>15</td><td>2</td></tr>
        </table>
      </div>
    </div>
    <!-- Right -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Geometric view
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Treat the row vector of a word as its representation: $\mathbf{v}_w = X_{w,:}$
        </div>
        <div style="margin-bottom:8px;">
          Compare two words by cosine similarity: $\cos(\mathbf{v},\mathbf{w}).$
        </div>
        <div>
          Words with similar count patterns will have similar vectors.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Intuition
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          <b>battle</b> has high counts in historical plays
        </div>
        <div>
          <b>fool</b> has high counts in comedies
        </div>
      </div>
      <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Counting-based methods represent a word by how often it appears with documents or contexts
  </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Counting-based: Raw Counts Are Not Enough</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        The problem with raw frequency
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Frequency is useful: if <b>sugar</b> often appears near <b>apricot</b>,
          that tells us something meaningful.
        </div>
        <div style="margin-bottom:8px;">
          But extremely common words like
          <b>the</b>, <b>it</b>, or <b>they</b>
          appear almost everywhere.
        </div>
        <div>
          So raw counts mix together
          <b>informativeness</b> and <b>global frequency</b>.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Desired effect
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        We want to
        <b>downweight frequent but uninformative words</b>
        and
        <b>upweight distinctive words</b>.
      </div>
    </div>
    <!-- Right -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Two standard fixes
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        <div style="margin-bottom:8px;">
          <b>TF-IDF</b> for term-document matrices
        </div>
        <div>
          \[
          w_{t,d} = \mathrm{TF}_{t,d}\cdot \mathrm{IDF}_t
          \]
          Frequent in one document but rare in the whole corpus
          \(\Rightarrow\) high weight
        </div>
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          <b>PPMI</b> for word-context matrices
        </div>
        <div>
          \[
          \mathrm{PMI}(w,c)=
          \log \frac{p(w,c)}{p(w)p(c)},
          \qquad
          \mathrm{PPMI}(w,c)=\max(\mathrm{PMI}(w,c),0)
          \]
          Upweight word-context pairs that occur together
          more often than chance
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Raw counts are a starting point; weighting makes them much more informative
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">TF-IDF for a Term-Document Matrix</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Definition
      </div>
      <div style="font-size:0.6em; line-height:1.44; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          <b>Term frequency</b>
          \[
          \mathrm{TF}_{t,d} = \log_{10}(\mathrm{count}(t,d)+1)
          \]
        </div>
        <div style="margin-bottom:8px;">
          <b>Document frequency</b>
          \[
          \mathrm{DF}_t = \#\{\text{documents containing } t\}
          \]
        </div>
        <div style="margin-bottom:8px;">
          <b>Inverse document frequency</b>
          \[
          \mathrm{IDF}_t = \log_{10}\!\left(\frac{N}{\mathrm{DF}_t}\right)
          \]
        </div>
        <div>
          <b>TF-IDF weight $w_{t,d} = \mathrm{TF}_{t,d}\cdot \mathrm{IDF}_t$.
        </div>
      </div>
    </div>
    <!-- Right -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Effect on example words
      </div>
      <div style="font-size:0.6em; line-height:1.35; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:10px 12px;">
        <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.95em;">
          <tr style="background:#4d79c7; color:white;">
            <th style="padding:4px;">word</th>
            <th style="padding:4px;">DF</th>
            <th style="padding:4px;">IDF</th>
            <th style="padding:4px;">interpretation</th>
          </tr>
          <tr>
            <td style="padding:4px; font-weight:700;">battle</td>
            <td>21</td>
            <td>0.246</td>
            <td>moderately distinctive</td>
          </tr>
          <tr style="background:#eef2fb;">
            <td style="padding:4px; font-weight:700;">wit</td>
            <td>34</td>
            <td>0.037</td>
            <td>fairly common</td>
          </tr>
          <tr>
            <td style="padding:4px; font-weight:700;">good</td>
            <td>37</td>
            <td>0</td>
            <td>too common, little value</td>
          </tr>
        </table>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Main takeaway
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          A word like <b>good</b> appears in almost every document,
          so its IDF is near zero.
        </div>
        <div>
          A more distinctive word like <b>battle</b> gets a larger weight,
          especially in documents where it appears often.
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    TF-IDF keeps local importance but discounts globally common words
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Counting-based: Co-occurrence Matrix</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Building the matrix
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Choose a context window around each target word \(w\), e.g. 4 words to the left and right.
        </div>
        <div style="margin-bottom:8px;">
          Build a matrix $X \in \mathbb{R}^{|V|\times |C|}$ where rows are target words and columns are context words.
        </div>
        <div>
          Each entry is a count: $X_{ij} = \#(w_i,\; c_j).$
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Word representation
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        The row vector of word \(w_i\) is its count-based representation: $\mathbf{v}_{w_i}=X_{i,:}.$
        Two words are similar if their context distributions are similar.
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Measuring similarity
      </div>
      <div style="font-size:0.6em; line-height:1.35; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:10px 12px; margin-bottom:12px;">
        <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.95em;">
          <tr style="background:#4d79c7; color:white;">
            <th style="padding:4px;">word</th>
            <th style="padding:4px;">computer</th>
            <th style="padding:4px;">data</th>
            <th style="padding:4px;">pie</th>
            <th style="padding:4px;">sugar</th>
          </tr>
          <tr>
            <td style="padding:4px; font-weight:700;">cherry</td>
            <td>2</td><td>8</td><td>442</td><td>25</td>
          </tr>
          <tr style="background:#eef2fb;">
            <td style="padding:4px; font-weight:700;">strawberry</td>
            <td>0</td><td>0</td><td>60</td><td>19</td>
          </tr>
          <tr>
            <td style="padding:4px; font-weight:700;">digital</td>
            <td>1670</td><td>1683</td><td>5</td><td>4</td>
          </tr>
          <tr style="background:#eef2fb;">
            <td style="padding:4px; font-weight:700;">information</td>
            <td>3325</td><td>3982</td><td>5</td><td>13</td>
          </tr>
        </table>
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Compare two row vectors by cosine similarity: $\cos(\mathbf{v},\mathbf{w})$
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          <b>cherry</b> and <b>strawberry</b> look similar because both occur with contexts like <b>pie</b> and <b>sugar</b>.
        </div>
        <div>
          <b>digital</b> and <b>information</b> look similar because both occur with <b>computer</b> and <b>data</b>.
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    key idea: two words are similar if their context-word vectors are similar
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Counting-based: PMI / PPMI Weighting</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Why raw counts are problematic
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Raw frequency mixes together <b>informativeness</b> and <b>global popularity</b>. Very common context words occur with almost everything, so large counts do not always mean a strong semantic relation.
        </div>
        <div>
          We want to know whether \(w\) and \(c\) co-occur
          <b>more than expected by chance</b>.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        PMI
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        Let $p_{ij}=p(w_i,c_j),\qquad p_{i*}=p(w_i),\qquad p_{*j}=p(c_j).$
        Then pointwise mutual information is
        \[
        \mathrm{PMI}(i,j)
        =
        \log \frac{p_{ij}}{p_{i*}\,p_{*j}}.
        \]
        A large PMI means \(w_i\) and \(c_j\) appear together more often than independence predicts.
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Positive PMI (PPMI)
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        PMI can be negative, which is often less useful in practice.
        So we keep only positive values:
        \[
        \mathrm{PPMI}(i,j)=\max(\mathrm{PMI}(i,j),\,0).
        \]
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        <div style="margin-bottom:6px;">
          <b>cherry–pie</b> should get a high PPMI
        </div>
        <div style="margin-bottom:6px;">
          <b>digital–computer</b> should get a high PPMI
        </div>
        <div>
          frequent but uninformative contexts get downweighted
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Limitation and bridge
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Even after PPMI weighting, the vectors are still typically
          <b>high-dimensional</b> and <b>sparse</b>.
        </div>
        <div>
          This motivates the next step:
          learn <b>short, dense embeddings</b> from these context statistics.
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="muted">Text Classification and Word Meaning</li>
    <li class="muted">Counting-based Methods</li>
    <li class="active">Learning-based Methods</li>
    <li class="muted">Bridge to LLMs</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Background and Core Idea</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Background
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          <b>word2vec</b> was introduced by Tomas Mikolov and collaborators in
          <b>2013</b> and quickly became one of the most influential methods
          for learning word embeddings.
        </div>
        <div style="margin-bottom:8px;">
          It marked a shift from <b>counting-based</b> representations
          to <b>prediction-based</b> representations.
        </div>
        <div>
          Instead of building a word vector directly from raw co-occurrence counts,
          word2vec <b>learns</b> vectors by solving an auxiliary prediction task.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Two famous 2013 papers
      </div>
      <div style="font-size:0.6em; line-height:1.38; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
  • <a href="https://baojian.github.io/llm-26/papers/lecture-03-readings-1-word2vec1.pdf" target="_blank"><i>Efficient Estimation of Word Representations ...</i></a>
</div>
<div>
  • <a href="https://baojian.github.io/llm-26/papers/lecture-03-readings-2-word2vec2.pdf" target="_blank"><i>Distributed Representations of Words and Phrases ...</i></a>
</div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Core idea
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Learn an embedding matrix $E \in \mathbb{R}^{|V|\times d}$ by training a simple model to predict nearby words.
        Given a target word \(w_t\), predict a context word \(w_{t+j}\)
          within a small window.
        <div style="text-align:center; margin:8px 0;">
          \[
          \text{target word} \;\longrightarrow\; \text{predict nearby words}
          \]
        </div>
        <div>
          Words that appear in similar contexts are pushed to have similar vectors.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Why it matters
      </div>
      <div style="font-size:0.6em; line-height:1.42; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          • <b>predict rather than count</b>
        </div>
        <div style="margin-bottom:6px;">
          • learn <b>dense, low-dimensional</b> word vectors
        </div>
        <div>
          • provides a simple and efficient framework for modern embedding learning
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    word2vec learns word meaning by predicting context, not by directly counting context
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Intuition</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:1.0;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        From counting to prediction
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Consider a running sentence:
        </div>
        <div style="text-align:center; font-size:1.05em; margin:10px 0 12px 0;">
          The <span style="color:#5a9f3d;">quick brown</span>
          <span style="color:#d62728; font-weight:700;">fox</span>
          <span style="color:#5a9f3d;">jumps over</span>
          the lazy dog
        </div>
        <div>
          Instead of counting how often each word appears near
          <span style="color:#d62728; font-weight:700;">fox</span>,
          word2vec trains a small model to <b>predict nearby words</b>.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Self-supervision
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        No human labels are needed.  
        If a word \(c\) occurs near the target word \(w\), then \((w,c)\) is automatically a useful training signal.
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Core idea
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        <div style="margin-bottom:8px;">
          <b>Target word</b> \(\rightarrow\) <b>predict context words</b>
        </div>
        <div style="text-align:center; margin:8px 0;">
          \[
          w_t \;\longrightarrow\; w_{t+j}, \qquad j \in \{-m,\dots,-1,1,\dots,m\}
          \]
        </div>
        <div>
          Words that occur in similar contexts will end up with similar vectors.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        What do we keep after training?
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        We are not mainly interested in the prediction accuracy itself.  
        We keep the learned parameters as the <b>word embeddings</b>.
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    word2vec learns word vectors by solving a self-supervised prediction task
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Constructing Training Pairs</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Positive pairs
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Let the target word be
          <span style="color:#d62728; font-weight:700;">fox</span>
          with window size \(m=2\):
        </div>
        <div style="text-align:center; font-size:1.02em; margin:10px 0 12px 0;">
          The
          <span style="color:#12a150;">quick brown</span>
          <span style="color:#d62728; font-weight:700;">fox</span>
          <span style="color:#12a150;">jumps over</span>
          the lazy dog
        </div>
        <div style="margin-bottom:8px;">
          Then the observed context words are:
          \[
          \{\text{quick},\text{brown},\text{jumps},\text{over}\}.
          \]
        </div>
        <div>
          So we create positive training pairs:
          \[
          (\text{fox},\text{quick}),\;
          (\text{fox},\text{brown}),\;
          (\text{fox},\text{jumps}),\;
          (\text{fox},\text{over}).
          \]
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Negative pairs
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        In addition to observed neighbors, word2vec samples
        <b>noise words</b> that did not occur in the context window, e.g.
        \[
        \text{aardvark},\; \text{seven},\; \text{forever},\; \text{dear}.
        \]
        This gives negative pairs such as
        \[
        (\text{fox},\text{aardvark}),\;
        (\text{fox},\text{seven}),\;
        (\text{fox},\text{forever}).
        \]
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Training view
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        it turns the corpus into a binary classification:
        <div style="text-align:center; margin-top:8px;">
          \[
          (w,c)\;\longmapsto\;
          \begin{cases}
          1, & \text{if } c \text{ is a real neighbor of } w \\
          0, & \text{if } c \text{ is a sampled noise word}
          \end{cases}
          \]
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Observed neighbors give positive pairs; sampled noise words give negative pairs
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Negative Sampling Objective</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Scoring a word-context pair
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:8px;">
          Let \(\mathbf{u}_w\) be the embedding of target word \(w\),
          and \(\mathbf{v}_c\) be the embedding of context word \(c\). Score the pair by the dot product: $s(w,c)=\mathbf{u}_w^\top \mathbf{v}_c.$
        </div>
        <div>
          Convert it to a probability by sigmoid:
          \[
          P(D=1\mid w,c)=\sigma(\mathbf{u}_w^\top \mathbf{v}_c),
          \]
          where \(D=1\) means “\(c\) is a true context word of \(w\)”.
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        Negative pair probability
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        \[
        P(D=0\mid w,c)=1-\sigma(\mathbf{u}_w^\top \mathbf{v}_c)
        =\sigma(-\mathbf{u}_w^\top \mathbf{v}_c).
        \]
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Loss for one positive pair
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Suppose \(c_{pos}\) is a real context word and
        \(c_{neg,1},\dots,c_{neg,k}\) are \(k\) sampled noise words.
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        The negative-sampling loss is
        \[
        L(w,c_{pos},c_{neg,1:k})
        =
        -\log \sigma(\mathbf{u}_w^\top \mathbf{v}_{c_{pos}})
        -
        \sum_{i=1}^{k}
        \log \sigma(-\mathbf{u}_w^\top \mathbf{v}_{c_{neg,i}}).
        \]
        <div style="margin-top:10px;">
          This objective:
          <div style="margin-top:6px;">
            • makes \(w\) similar to true context words  
          </div>
          <div>
            • makes \(w\) dissimilar to sampled noise words
          </div>
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Maximize similarity to true neighbors, minimize similarity to noise samples
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Model Parameters</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <!-- Left column -->
    <div style="flex:0.82;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        What parameters are learned?
      </div>
      <div style="font-size:0.6em; line-height:1.44; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        word2vec learns two parameter matrices:
        \[
        \theta = (W,\;C)
        \]
        where $W \in \mathbb{R}^{|V_w|\times d}, \qquad C \in \mathbb{R}^{|V_c|\times d}.$
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Interpretation
      </div>
      <div style="font-size:0.6em; line-height:1.44; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          • \(W\): <b>target-word</b> embedding matrix
        </div>
        <div style="margin-bottom:6px;">
          • \(C\): <b>context-word</b> embedding matrix
        </div>
        <div>
          • \(|V_w|\)</b>: target-word vocabulary size &nbsp;&nbsp;
        </div>
        <div>
          • \(|V_c|\)</b>: context-word vocabulary size
        </div>
        <div>
          • \(d\)</b>: embedding dimension &nbsp;&nbsp;
        </div>
        <div>
          • \(k\)</b>: number of negative samples
        </div>
      </div>
    </div>
    <!-- Right column -->
    <div style="flex:1.18;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Matrix view
      </div>
      <div style="display:flex; gap:14px; font-size:0.5em; line-height:1.3;">
        <div style="flex:1; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:10px 12px;">
          <div style="text-align:center; font-weight:800; color:#1f4ba5; margin-bottom:6px;">
            \(W\): target words
          </div>
          <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.95em;">
            <tr style="background:#11b25c; color:white;">
              <th style="padding:4px;">the</th><th>2</th><th>0.5</th><th>\(\cdots\)</th><th>4</th>
            </tr>
            <tr style="background:#1ea7e1; color:white;">
              <th style="padding:4px;">quick</th><td>0.1</td><td>0.2</td><td></td><td>0.7</td>
            </tr>
            <tr style="background:#4d79c7; color:white;">
              <th style="padding:4px;">brown</th><td>0.3</td><td>-2</td><td></td><td>0.2</td>
            </tr>
            <tr style="background:#f2b500; color:black;">
              <th style="padding:4px;">fox</th><td>-2</td><td>0.2</td><td></td><td>0.8</td>
            </tr>
            <tr><th style="padding:6px;">\(\vdots\)</th><td>\(\vdots\)</td><td>\(\vdots\)</td><td></td><td>\(\vdots\)</td></tr>
            <tr style="background:#69a8df; color:white;">
              <th style="padding:4px;">lazy</th><td>1</td><td>0.7</td><td></td><td>3</td>
            </tr>
            <tr style="background:#7dbb42; color:white;">
              <th style="padding:4px;">dog</th><td>3</td><td>5</td><td></td><td>0.2</td>
            </tr>
          </table>
        </div>
        <div style="flex:1; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:10px 12px;">
          <div style="text-align:center; font-weight:800; color:#1f4ba5; margin-bottom:6px;">
            \(C\): context words
          </div>
          <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.95em;">
            <tr style="background:#11b25c; color:white;">
              <th style="padding:4px;">the</th><th>1.5</th><th>0.2</th><th>\(\cdots\)</th><th>3</th>
            </tr>
            <tr style="background:#1ea7e1; color:white;">
              <th style="padding:4px;">quick</th><td>0.9</td><td>0.3</td><td></td><td>0.7</td>
            </tr>
            <tr style="background:#4d79c7; color:white;">
              <th style="padding:4px;">brown</th><td>-2</td><td>1</td><td></td><td>0.7</td>
            </tr>
            <tr style="background:#f2b500; color:black;">
              <th style="padding:4px;">fox</th><td>3</td><td>0.4</td><td></td><td>0.9</td>
            </tr>
            <tr><th style="padding:6px;">\(\vdots\)</th><td>\(\vdots\)</td><td>\(\vdots\)</td><td></td><td>\(\vdots\)</td></tr>
            <tr style="background:#69a8df; color:white;">
              <th style="padding:4px;">lazy</th><td>3</td><td>0.6</td><td></td><td>2.5</td>
            </tr>
            <tr style="background:#7dbb42; color:white;">
              <th style="padding:4px;">dog</th><td>7</td><td>2</td><td></td><td>0.1</td>
            </tr>
          </table>
        </div>
      </div>
      <div style="margin-top:12px; font-size:0.6em; line-height:1.42; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        In neural-network notation, these are often written as
        \[
        W = W_{\text{in}}, \qquad C = W_{\text{out}}.
        \]
        After training, we usually keep \(W\), or average \(W\) and \(C\), as the final word embeddings.
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Running Example Setup</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Training corpus
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Consider the sentence
        <div style="text-align:center; font-size:1.05em; margin:10px 0 4px 0;">
          the quick brown fox jumps over the lazy dog
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Model settings
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          • window size \(m=2\)
        </div>
        <div style="margin-bottom:6px;">
          • embedding dimension \(d=3\)
        </div>
        <div style="margin-bottom:6px;">
          • vocabulary
          \[
          V=\{\text{the, quick, brown, fox, jumps, over, lazy, dog}\}
          \]
        </div>
        <div>
          • vocabulary size \(|V|=8\)
        </div>
      </div>
    </div>
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        One training position
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Take
        <span style="color:#1ea7e1; font-weight:700;">quick</span>
        as the target word:
        <div style="text-align:center; font-size:1.0em; margin:10px 0;">
          the
          <span style="color:#1ea7e1; font-weight:700;">quick</span>
          brown fox jumps over the lazy dog
        </div>
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        With window size \(m=2\), the observed context words are
        \[
        \{\text{the},\text{brown},\text{fox}\}.
        \]
        So the positive pairs are
        \[
        (\text{quick},\text{the}),\;
        (\text{quick},\text{brown}),\;
        (\text{quick},\text{fox}).
        \]
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    word2vec turns the corpus into many target-context training pairs
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: From One-Hot Input to Hidden Vector</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.95;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Input representation
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        The target word
        <span style="color:#1ea7e1; font-weight:700;">quick</span>
        is represented by a one-hot vector $\mathbf{x}\in\mathbb{R}^{|V|}.$ If the vocabulary order is
        \[
        (\text{the, quick, brown, fox, jumps, over, lazy, dog}),
        \]
        then $\mathbf{x}_{\text{quick}}= [0,1,0,0,0,0,0,0]^\top.$
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Input embedding matrix
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        Let $W_{\text{in}}\in\mathbb{R}^{|V|\times d}.$ Each row of \(W_{\text{in}}\) is the embedding of one target word.
      </div>
    </div>
    <div style="flex:1.05;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Projection = embedding lookup
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        The hidden vector is $\mathbf{h}=W_{\text{in}}^\top \mathbf{x}.$ Since \(\mathbf{x}\) is one-hot, this simply selects the row for
        <span style="color:#1ea7e1; font-weight:700;">quick</span>.
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        Example: $W_{\text{in}}[\text{quick}] = [0.1,\;0.2,\;0.7].$ Therefore $\mathbf{h}=[0.1,\;0.2,\;0.7]^\top.$
      </div>
      <div style="margin-top:12px; font-size:0.7em; line-height:1.42; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        So the “hidden layer” in skip-gram is not complicated:
        it is just the embedding vector of the target word.
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    One-hot input + embedding matrix = lookup the target word vector
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Output Layer and Softmax Prediction</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Output embedding matrix
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Let $W_{\text{out}}\in\mathbb{R}^{|V|\times d}.$ Each row \(W_{\text{out}}[c]\) is a vector for a context word \(c\).
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Score each possible context word
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        For each candidate context word \(c\), compute
        \[
        s_c = W_{\text{out}}[c]^\top \mathbf{h}.
        \]
        Then apply softmax: $P(c\mid w) = \frac{\exp(s_c)} {\sum_{c'\in V}\exp(s_{c'})}.$
      </div>
    </div>
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Running example
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Given target word
        <span style="color:#1ea7e1; font-weight:700;">quick</span>,
        the model produces a probability distribution over all 8 vocabulary words:
        \[
        P(\text{the}\mid \text{quick}),\;
        P(\text{brown}\mid \text{quick}),\;
        P(\text{fox}\mid \text{quick}),\dots
        \]
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        We want real neighbors such as
        <b>the</b>, <b>brown</b>, and <b>fox</b>
        to get high probability.
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <b>Problem:</b> softmax needs scores for
        <b>every word in the vocabulary</b>.
        For large vocabularies, this is expensive.
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Full softmax is conceptually simple, but computationally expensive
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Negative Sampling</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Positive pair
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Pick one observed context word, for example
        \[
        (\text{quick},\text{brown}).
        \]
        This is a positive pair because
        <b>brown</b> appears near <b>quick</b> in the corpus.
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Negative samples
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        Instead of comparing against the whole vocabulary,
        sample a few noise words, e.g. $\text{dog},\; \text{lazy},\; \text{jumps}.$
        Then we only distinguish $(\text{quick},\text{brown})$ from
        \[
        (\text{quick},\text{dog}),\;
        (\text{quick},\text{lazy}),\;
        (\text{quick},\text{jumps}).
        \]
      </div>
    </div>
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Binary objective
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Score a pair by $s(w,c)=W_{\text{in}}[w]^\top W_{\text{out}}[c].$
        Turn it into a probability by sigmoid:
        \[
        P(D=1\mid w,c)=\sigma(s(w,c)).
        \]
      </div>
      <div style="font-size:0.6em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        For one positive pair and \(k\) negatives, minimize
        \[
        L
        =
        -\log \sigma(s(w,c_{pos}))
        -
        \sum_{i=1}^{k}\log \sigma(-s(w,c_{neg,i})).
        \]
        <div style="margin-top:10px;">
          This makes
          <b>(quick, brown)</b> similar,
          while pushing
          <b>(quick, dog)</b>, <b>(quick, lazy)</b>, ...
          apart.
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Negative sampling replaces expensive full softmax by a small binary classification problem
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Quick Summary</div>
  <div class="ppt-line"></div>

  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Positive vs. negative pairs
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        Assume the center word is
        <span style="color:#d62728; font-weight:700;">regression</span>.
        <div style="margin-top:8px; margin-bottom:6px;">
          Likely context words:
          <b>logistic</b>, <b>machine</b>, <b>sigmoid</b>, <b>supervised</b>, <b>neural</b>
        </div>
        <div>
          Unlikely sampled words:
          <b>zebra</b>, <b>pimples</b>, <b>toothpaste</b>, <b>idiot</b>
        </div>
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin:12px 0 8px 0;">
        What the model tries to do
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          • maximize \(P(D=1\mid w,c_{pos})\) for positive pairs
        </div>
        <div>
          • minimize \(P(D=1\mid w,c_{neg})\) for negative pairs
        </div>
      </div>
    </div>
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Core intuition
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        If the model can distinguish
        <b>real target-context pairs</b>
        from
        <b>noise pairs</b>,
        then useful word vectors will be learned.
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        After training:
        <div style="margin-top:8px;">
          • words with similar neighborhoods get similar embeddings
        </div>
        <div>
          • semantically related words tend to be close in vector space
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    word2vec learns embeddings by turning context prediction into a binary classification problem.
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: Training Objective and SGD</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Parameters
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        The model parameters are the two embedding matrices: $\theta=(W_{\text{in}}, W_{\text{out}}).$ For one training case, only the current target vector and a few context vectors are involved.
      </div>
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Local negative-sampling loss
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        For target vector \(\mathbf{w}\), one positive context \(\mathbf{c}_{pos}\),
        and \(k\) negative contexts \(\mathbf{c}_{neg,1},\dots,\mathbf{c}_{neg,k}\),
        define
        \[
        \ell(\theta)
        =
        -\log \sigma(\mathbf{c}_{pos}^{\top}\mathbf{w})
        -
        \sum_{i=1}^{k}\log \sigma(-\mathbf{c}_{neg,i}^{\top}\mathbf{w}).
        \]
      </div>
    </div>
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Corpus-level objective and SGD update
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        Over all training positions, minimize the average loss:
        \[
        J(\theta)=\frac{1}{T}\sum_{t=1}^{T}\ell_t(\theta).
        \]
        Update parameters using one training case at a time:
        \[
        \theta^{t+1}
        =
        \theta^{t}
        -
        \eta_t \nabla \ell_t(\theta^{t}),
        \]
        where \(\eta_t\) is the learning rate.
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          • positive pairs pull embeddings closer
        </div>
        <div>
          • negative pairs push embeddings apart
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    Train by minimizing a local binary-classification loss with SGD
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">word2vec: One SGD Step for Parameter Updates</div>
  <div class="ppt-line"></div>
  <div class="ppt-body" style="display:flex; gap:28px; align-items:flex-start; margin-top:16px;">
    <div style="flex:0.98;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        Gradients
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#f7f8fc; border:1px solid #d9deea; border-radius:12px; padding:12px 14px;">
        Let $s_{pos}=\sigma(\mathbf{c}_{pos}^{\top}\mathbf{w}), s_i=\sigma(\mathbf{c}_{neg,i}^{\top}\mathbf{w}).$
        Then
        \[
        \frac{\partial \ell}{\partial \mathbf{c}_{pos}}=(s_{pos}-1)\mathbf{w},
        \qquad
        \frac{\partial \ell}{\partial \mathbf{c}_{neg,i}}=s_i\,\mathbf{w},
        \]
        \[
        \frac{\partial \ell}{\partial \mathbf{w}}
        =
        (s_{pos}-1)\mathbf{c}_{pos}
        +
        \sum_{i=1}^{k} s_i\,\mathbf{c}_{neg,i}.
        \]
      </div>
    </div>
    <div style="flex:1.02;">
      <div style="font-size:0.8em; font-weight:800; color:#1f4ba5; margin-bottom:8px;">
        SGD updates
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fbfbfd; border:1px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        \[
        \mathbf{c}_{pos}
        \leftarrow
        \mathbf{c}_{pos}
        -
        \eta_t (s_{pos}-1)\mathbf{w},
        \]
        \[
        \mathbf{c}_{neg,i}
        \leftarrow
        \mathbf{c}_{neg,i}
        -
        \eta_t s_i\,\mathbf{w},
        \qquad i=1,\dots,k,
        \]
        \[
        \mathbf{w}
        \leftarrow
        \mathbf{w}
        -
        \eta_t
        \left[
        (s_{pos}-1)\mathbf{c}_{pos}
        +
        \sum_{i=1}^{k} s_i\,\mathbf{c}_{neg,i}
        \right].
        \]
      </div>
      <div style="font-size:0.7em; line-height:1.45; background:#fff8f0; border:1px solid #eed9b6; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;">
          <b>Effect:</b>
        </div>
        <div style="margin-bottom:6px;">
          • move \(\mathbf{w}\) closer to the positive context vector
        </div>
        <div>
          • move \(\mathbf{w}\) away from the negative context vectors
        </div>
      </div>
    </div>
  </div>
  <div style="margin-top:14px; text-align:center; font-size:0.78em; font-weight:700; color:#1f4ba5;">
    One update step pulls target and true context together, and pushes target away from noise samples
  </div>
</section>

---

<!-- .slide: class="ppt" -->
## Training word2vec <span style="color:#1f77b4; font-size:0.75em;">running example setup</span>

- Example sentence: **Ned Stark is the most honorable man**
- Target word: <span style="color:#63a33b;"><b>Ned</b></span>
- Positive context word: <span style="color:#e67e22;"><b>Stark</b></span>
- Negative samples:
  <span style="background:#fff3a3; padding:0 0.15em;">pimples</span>,
  <span style="background:#fff3a3; padding:0 0.15em;">zebra</span>,
  <span style="background:#fff3a3; padding:0 0.15em;">idiot</span>

- Input embedding matrix: \(W_{\text{in}} \in \mathbb{R}^{|V|\times d}\)
- Output embedding matrix: \(W_{\text{out}} \in \mathbb{R}^{|V|\times d}\)

<div style="margin-top:0.8em; padding:0.7em 0.9em; border:1px solid #d9deea; border-radius:12px; background:#f7f8fc; font-size:0.9em;">
For one training case, we only update:
\[
\mathbf{w}_{\text{Ned}},\quad
\mathbf{c}_{\text{Stark}},\quad
\mathbf{c}_{\text{pimples}},\quad
\mathbf{c}_{\text{zebra}},\quad
\mathbf{c}_{\text{idiot}}.
\]
</div>

<div style="margin-top:0.9em; text-align:center; font-weight:700; color:#1f4ba5;">
One positive pair + a few negative pairs
</div>

---

<!-- .slide: class="ppt" -->
## Training word2vec <span style="color:#1f77b4; font-size:0.75em;">forward pass</span>

- One-hot input for <span style="color:#63a33b;"><b>Ned</b></span> selects its row from \(W_{\text{in}}\):
\[
\mathbf{h} = \mathbf{w}_{\text{Ned}} = [-0.018,\;0.404,\;-0.317]^\top
\]

- Score each candidate context word by dot product:
\[
s_j = \mathbf{c}_j^\top \mathbf{h}
\]

- Convert score to probability by sigmoid:
\[
p_j = \sigma(s_j)
\]

<div style="margin-top:0.7em; font-size:0.9em;">

| Pair | Score \(s_j\) | Probability \(\sigma(s_j)\) | Label \(t_j\) |
|---|---:|---:|---:|
| \((\text{Ned}, \text{Stark})\) | \(0.508\) | \(0.624\) | \(1\) |
| \((\text{Ned}, \text{pimples})\) | \(0.213\) | \(0.553\) | \(0\) |
| \((\text{Ned}, \text{zebra})\) | \(0.136\) | \(0.534\) | \(0\) |
| \((\text{Ned}, \text{idiot})\) | \(-0.132\) | \(0.467\) | \(0\) |

</div>

<div style="margin-top:0.7em; padding:0.7em 0.9em; border:1px solid #eed9b6; border-radius:12px; background:#fff8f0; font-size:0.9em;">
Loss for this training case:
\[
J
=
-\log \sigma(\mathbf{c}_{pos}^{\top}\mathbf{h})
-\sum_{i=1}^{3}\log \sigma(-\mathbf{c}_{neg_i}^{\top}\mathbf{h})
\]
</div>

---

<!-- .slide: class="ppt" -->
## Training word2vec <span style="color:#1f77b4; font-size:0.75em;">backward pass: prediction error and \(\nabla \mathbf{w}_{in}\)</span>

- Prediction error for each selected output row:
\[
\delta_j = \sigma(\mathbf{c}_j^\top \mathbf{h}) - t_j
\]

<div style="margin-top:0.6em; font-size:0.92em;">

| Word \(j\) | \(\sigma(\mathbf{c}_j^\top \mathbf{h})\) | \(t_j\) | \(\delta_j\) |
|---|---:|---:|---:|
| Stark | \(0.624\) | \(1\) | \(-0.376\) |
| pimples | \(0.553\) | \(0\) | \(0.553\) |
| zebra | \(0.534\) | \(0\) | \(0.534\) |
| idiot | \(0.467\) | \(0\) | \(0.467\) |

</div>

- Gradient w.r.t. the target embedding:
\[
\frac{\partial J}{\partial \mathbf{w}}
=
(\sigma(\mathbf{c}_{pos}^\top \mathbf{w})-1)\mathbf{c}_{pos}
+
\sum_{i=1}^{3}\sigma(\mathbf{c}_{neg_i}^\top \mathbf{w})\mathbf{c}_{neg_i}
\]

- For this example:
\[
\nabla \mathbf{w}_{in}
=
[-0.932,\;0.319,\;0.655]^\top
\]

<div style="margin-top:0.8em; padding:0.7em 0.9em; border:1px solid #d9deea; border-radius:12px; background:#f7f8fc; font-size:0.9em;">
Interpretation: the target vector <b>Ned</b> is pushed
toward <b>Stark</b> and away from the negative samples.
</div>

---

<!-- .slide: class="ppt" -->
## Training word2vec <span style="color:#1f77b4; font-size:0.75em;">backward pass: \(\nabla \mathbf{w}_{out}\)</span>

- Positive context gradient:
\[
\frac{\partial J}{\partial \mathbf{c}_{pos}}
=
(\sigma(\mathbf{c}_{pos}^\top \mathbf{w})-1)\mathbf{w}
\]

- Negative context gradients:
\[
\frac{\partial J}{\partial \mathbf{c}_{neg_i}}
=
\sigma(\mathbf{c}_{neg_i}^\top \mathbf{w})\mathbf{w}
\]

<div style="margin-top:0.7em; font-size:0.9em;">

| Updated row in \(W_{out}\) | Gradient |
|---|---|
| \(\nabla \mathbf{c}_{\text{Stark}}\) | \([\,0.007,\;-0.152,\;0.119\,]\) |
| \(\nabla \mathbf{c}_{\text{pimples}}\) | \([\, -0.010,\;0.223,\;-0.175\,]\) |
| \(\nabla \mathbf{c}_{\text{zebra}}\) | \([\, -0.010,\;0.216,\;-0.169\,]\) |
| \(\nabla \mathbf{c}_{\text{idiot}}\) | \([\, -0.008,\;0.189,\;-0.148\,]\) |

</div>

<div style="margin-top:0.8em; padding:0.7em 0.9em; border:1px solid #eed9b6; border-radius:12px; background:#fff8f0; font-size:0.9em;">
Only a <b>few rows</b> of \(W_{out}\) are touched in one SGD step:
one positive context row and a few sampled negative rows.
</div>

---

<!-- .slide: class="ppt" -->
## Training word2vec <span style="color:#1f77b4; font-size:0.75em;">SGD parameter update</span>

- With learning rate \(\eta = 0.05\), update by
\[
\theta \leftarrow \theta - \eta \nabla_\theta J
\]

### Update target row in \(W_{in}\)

\[
\mathbf{w}_{\text{Ned}}^{old}
=
[-0.018,\;0.404,\;-0.317]
\]

\[
\mathbf{w}_{\text{Ned}}^{new}
=
\mathbf{w}_{\text{Ned}}^{old}
-
0.05 \cdot [-0.932,\;0.319,\;0.655]
=
[0.029,\;0.388,\;-0.350]
\]

### Update selected rows in \(W_{out}\)

\[
\mathbf{c}_{\text{Stark}}^{new}
=
[0.116,\;0.731,\;-0.695]
\]

\[
\mathbf{c}_{\text{pimples}}^{new}
=
[-0.940,\;0.590,\;0.155]
\]

\[
\mathbf{c}_{\text{zebra}}^{new}
=
[-0.622,\;0.800,\;0.648]
\]

\[
\mathbf{c}_{\text{idiot}}^{new}
=
[-0.077,\;-0.384,\;-0.049]
\]

<div style="margin-top:0.8em; padding:0.7em 0.9em; border:1px solid #d9deea; border-radius:12px; background:#f7f8fc; font-size:0.9em;">
<b>Takeaway:</b> one SGD step
pulls <b>Ned</b> closer to <b>Stark</b>,
and pushes it away from the sampled negative words.
</div>