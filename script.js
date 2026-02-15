// Show algorithm details with code examples and interactive demo
function showAlgorithmDetails(categoryKey, algorithmName) {
    const category = mlData[categoryKey];
    const algorithm = category.algorithms.find(a => a.name === algorithmName);
    const colors = colorMap[category.color];
    const modal = document.getElementById('algorithmModal');
    
    document.getElementById('modalTitle').textContent = algorithm.name;
    document.getElementById('modalCategory').textContent = category.title;
    
    // Check if code example exists
    const codeExample = window.codeExamples && window.codeExamples[algorithm.name];
    
    let content = `
        <div class="space-y-6">
            <!-- Tabs -->
            <div class="tab-container">
                <button class="tab-btn active" onclick="switchTab(event, 'overview')">
                    <i class="fas fa-info-circle mr-2"></i>Overview
                </button>
                ${codeExample ? `
                <button class="tab-btn" onclick="switchTab(event, 'code')">
                    <i class="fas fa-code mr-2"></i>Code Example
                </button>
                ` : ''}
                ${getInteractiveDemoTypes(algorithmName).length > 0 ? `
                <button class="tab-btn" onclick="switchTab(event, 'demo')">
                    <i class="fas fa-play-circle mr-2"></i>Try It Live
                </button>
                ` : ''}
            </div>
            
            <!-- Overview Tab -->
            <div id="overview" class="tab-content active">
                <div>
                    <h3 class="text-xl font-bold ${colors.text} mb-3">
                        <i class="fas fa-info-circle mr-2"></i>What It Does
                    </h3>
                    <p class="text-gray-700 leading-relaxed">${algorithm.description}</p>
                </div>
                
                <div class="${colors.bg} rounded-lg p-5">
                    <h3 class="text-xl font-bold ${colors.text} mb-3">
                        <i class="fas fa-lightbulb mr-2"></i>Real-World Examples
                    </h3>
                    <ul class="space-y-2">
                        ${algorithm.useCases.map(useCase => `
                            <li class="flex items-start">
                                <i class="fas fa-check-circle ${colors.text} mt-1 mr-3"></i>
                                <span class="text-gray-700">${useCase}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                
                <div class="grid md:grid-cols-2 gap-4">
                    <div class="bg-green-50 rounded-lg p-4">
                        <h4 class="font-bold text-green-700 mb-3">
                            <i class="fas fa-thumbs-up mr-2"></i>Strengths
                        </h4>
                        <ul class="space-y-1">
                            ${algorithm.strengths.map(strength => `
                                <li class="text-sm text-gray-700">
                                    <i class="fas fa-plus text-green-600 mr-2"></i>${strength}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    
                    <div class="bg-red-50 rounded-lg p-4">
                        <h4 class="font-bold text-red-700 mb-3">
                            <i class="fas fa-exclamation-triangle mr-2"></i>Limitations
                        </h4>
                        <ul class="space-y-1">
                            ${algorithm.limitations.map(limitation => `
                                <li class="text-sm text-gray-700">
                                    <i class="fas fa-minus text-red-600 mr-2"></i>${limitation}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            
            ${codeExample ? `
            <!-- Code Tab -->
            <div id="code" class="tab-content">
                <h3 class="text-xl font-bold ${colors.text} mb-4">
                    <i class="fas fa-code mr-2"></i>Python Implementation
                </h3>
                
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-lang">
                            <i class="fab fa-python mr-2"></i>Python
                        </span>
                        <button class="copy-btn" onclick="copyCode(this, 'code-${algorithmName.replace(/\s+/g, '-')}')">
                            <i class="fas fa-copy mr-1"></i>Copy
                        </button>
                    </div>
                    <pre><code class="language-python" id="code-${algorithmName.replace(/\s+/g, '-')}">${codeExample.python}</code></pre>
                </div>
                
                <div class="code-explanation">
                    <p><i class="fas fa-lightbulb mr-2"></i><strong>How it works:</strong> ${codeExample.explanation}</p>
                </div>
                
                <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                    <h4 class="font-bold text-blue-800 mb-2">
                        <i class="fas fa-rocket mr-2"></i>Try it yourself!
                    </h4>
                    <p class="text-sm text-blue-700">
                        Copy this code and run it in your Python environment. You'll need scikit-learn installed: 
                        <code class="bg-blue-100 px-2 py-1 rounded">pip install scikit-learn</code>
                    </p>
                </div>
            </div>
            ` : ''}
            
            ${getInteractiveDemoTypes(algorithmName).length > 0 ? `
            <!-- Interactive Demo Tab -->
            <div id="demo" class="tab-content">
                ${generateInteractiveDemo(algorithmName, categoryKey)}
            </div>
            ` : ''}
            
            <div class="text-center pt-4 border-t">
                <button onclick="showCategoryDetails('${categoryKey}')" 
                        class="${colors.bg} ${colors.text} px-6 py-2 rounded-lg font-semibold hover:shadow-md transition">
                    <i class="fas fa-arrow-left mr-2"></i>Back to ${category.title}
                </button>
            </div>
        </div>
    `;
    
    document.getElementById('modalContent').innerHTML = content;
    modal.style.display = 'block';
    
    // Highlight code syntax
    if (codeExample && typeof hljs !== 'undefined') {
        setTimeout(() => {
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }, 100);
    }
}

// Switch between tabs
function switchTab(event, tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Add active class to clicked tab
    event.currentTarget.classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

// Copy code to clipboard
function copyCode(button, codeId) {
    const code = document.getElementById(codeId).textContent;
    navigator.clipboard.writeText(code).then(() => {
        button.innerHTML = '<i class="fas fa-check mr-1"></i>Copied!';
        button.classList.add('copied');
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy mr-1"></i>Copy';
            button.classList.remove('copied');
        }, 2000);
    });
}

// Get interactive demo types for algorithm
function getInteractiveDemoTypes(algorithmName) {
    const demos = {
        "OLS (Ordinary Least Squares)": ["regression"],
        "Logistic Regression": ["classification"],
        "K-Means": ["clustering"],
    };
    return demos[algorithmName] || [];
}

// Generate interactive demo
function generateInteractiveDemo(algorithmName, categoryKey) {
    const colors = colorMap[mlData[categoryKey].color];
    
    if (algorithmName === "OLS (Ordinary Least Squares)") {
        return `
            <div class="demo-container">
                <h3 class="text-2xl font-bold mb-4">
                    <i class="fas fa-play-circle mr-2"></i>Try Linear Regression
                </h3>
                <p class="mb-6 text-white/90">Predict house prices based on size!</p>
                
                <div class="demo-input-group">
                    <label class="block font-semibold mb-2">House Size (sq ft)</label>
                    <input type="number" id="houseSize" class="demo-input" placeholder="Enter size (e.g., 1800)" value="1800">
                </div>
                
                <button class="demo-btn" onclick="runLinearRegressionDemo()">
                    <i class="fas fa-calculator mr-2"></i>Calculate Price
                </button>
                
                <div id="regressionResult" style="display: none;" class="demo-result">
                    <h4><i class="fas fa-chart-line mr-2"></i>Prediction Result</h4>
                    <div class="demo-result-value" id="predictedPrice"></div>
                    <p class="text-sm text-white/80 mt-2" id="regressionFormula"></p>
                </div>
            </div>
        `;
    }
    
    if (algorithmName === "Logistic Regression") {
        return `
            <div class="demo-container">
                <h3 class="text-2xl font-bold mb-4">
                    <i class="fas fa-play-circle mr-2"></i>Try Spam Detection
                </h3>
                <p class="mb-6 text-white/90">Check if an email is spam!</p>
                
                <div class="demo-input-group">
                    <label class="block font-semibold mb-2">Word Count</label>
                    <input type="number" id="wordCount" class="demo-input" placeholder="Number of words" value="150">
                </div>
                
                <div class="demo-input-group">
                    <label class="block font-semibold mb-2">Contains "Urgent"?</label>
                    <select id="hasUrgent" class="demo-input">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                
                <div class="demo-input-group">
                    <label class="block font-semibold mb-2">Contains "$$$"?</label>
                    <select id="hasMoney" class="demo-input">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                
                <button class="demo-btn" onclick="runSpamDetectionDemo()">
                    <i class="fas fa-search mr-2"></i>Check Email
                </button>
                
                <div id="spamResult" style="display: none;" class="demo-result">
                    <h4><i class="fas fa-envelope mr-2"></i>Detection Result</h4>
                    <div class="demo-result-value" id="spamVerdict"></div>
                    <p class="text-sm text-white/80 mt-2" id="spamConfidence"></p>
                </div>
            </div>
        `;
    }
    
    if (algorithmName === "K-Means") {
        return `
            <div class="demo-container">
                <h3 class="text-2xl font-bold mb-4">
                    <i class="fas fa-play-circle mr-2"></i>Try Customer Segmentation
                </h3>
                <p class="mb-6 text-white/90">Find which customer group you belong to!</p>
                
                <div class="demo-input-group">
                    <label class="block font-semibold mb-2">Age</label>
                    <input type="number" id="customerAge" class="demo-input" placeholder="Your age" value="30">
                </div>
                
                <div class="demo-input-group">
                    <label class="block font-semibold mb-2">Annual Spending ($)</label>
                    <input type="number" id="customerSpending" class="demo-input" placeholder="Annual spending" value="50000">
                </div>
                
                <button class="demo-btn" onclick="runKMeansDemo()">
                    <i class="fas fa-users mr-2"></i>Find My Segment
                </button>
                
                <div id="kmeansResult" style="display: none;" class="demo-result">
                    <h4><i class="fas fa-users mr-2"></i>Your Customer Segment</h4>
                    <div class="demo-result-value" id="customerSegment"></div>
                    <p class="text-sm text-white/80 mt-2" id="segmentDescription"></p>
                </div>
            </div>
        `;
    }
    
    return '<p class="text-gray-600">Interactive demo coming soon!</p>';
}

// Interactive Demo Functions
function runLinearRegressionDemo() {
    const size = parseFloat(document.getElementById('houseSize').value);
    
    // Simple linear model: Price = 200 * Size + 50000
    const slope = 200;
    const intercept = 50000;
    const price = slope * size + intercept;
    
    document.getElementById('regressionResult').style.display = 'block';
    document.getElementById('predictedPrice').textContent = `$${price.toLocaleString()}`;
    document.getElementById('regressionFormula').textContent = 
        `Formula: Price = ${slope} × Size + ${intercept.toLocaleString()}`;
}

function runSpamDetectionDemo() {
    const wordCount = parseInt(document.getElementById('wordCount').value);
    const hasUrgent = parseInt(document.getElementById('hasUrgent').value);
    const hasMoney = parseInt(document.getElementById('hasMoney').value);
    
    // Simple spam score calculation
    let spamScore = 0;
    if (wordCount > 100) spamScore += 30;
    if (hasUrgent === 1) spamScore += 40;
    if (hasMoney === 1) spamScore += 30;
    
    const isSpam = spamScore > 50;
    const confidence = Math.min(spamScore + 20, 95);
    
    document.getElementById('spamResult').style.display = 'block';
    document.getElementById('spamVerdict').textContent = isSpam ? '⚠️ SPAM' : '✅ NOT SPAM';
    document.getElementById('spamVerdict').style.color = isSpam ? '#ef4444' : '#10b981';
    document.getElementById('spamConfidence').textContent = `Confidence: ${confidence}%`;
}

function runKMeansDemo() {
    const age = parseInt(document.getElementById('customerAge').value);
    const spending = parseInt(document.getElementById('customerSpending').value);
    
    // Simple clustering logic
    let segment, description;
    
    if (age < 35 && spending < 40000) {
        segment = "🌱 Young Savers";
        description = "Budget-conscious millennials building their future";
    } else if (age < 35 && spending >= 40000) {
        segment = "🚀 Young Professionals";
        description = "High-earning millennials who love premium products";
    } else if (age >= 35 && spending < 60000) {
        segment = "🏡 Family Focused";
        description = "Practical spenders prioritizing family needs";
    } else {
        segment = "💎 Premium Customers";
        description = "High-value customers seeking luxury and quality";
    }
    
    document.getElementById('kmeansResult').style.display = 'block';
    document.getElementById('customerSegment').textContent = segment;
    document.getElementById('segmentDescription').textContent = description;
}
// Machine Learning Algorithms Database
const mlData = {
    regression: {
        title: "Regression",
        icon: "fa-chart-line",
        color: "blue",
        description: "When you need to predict numbers - like prices, temperatures, or sales figures",
        algorithms: [
            {
                name: "OLS (Ordinary Least Squares)",
                description: "The simplest form of regression - it finds the straight line that best fits your data points. Think of it like drawing the best-fit line through a scatter plot.",
                useCases: [
                    "Housing price prediction based on features like size, location, and amenities",
                    "Sales forecasting using historical data and marketing spend",
                    "Economic modeling to understand relationships between variables"
                ],
                strengths: ["Simple and interpretable", "Fast computation", "Works well with linear relationships"],
                limitations: ["Assumes linear relationships", "Sensitive to outliers", "Requires independence of errors"]
            },
            {
                name: "Lasso Regression",
                description: "A smart version of linear regression that automatically picks the most important features and ignores the rest. Really useful when you have tons of variables.",
                useCases: [
                    "Gene selection in bioinformatics with thousands of features",
                    "Text classification with sparse feature sets",
                    "Financial modeling where feature selection is crucial"
                ],
                strengths: ["Automatic feature selection", "Handles high-dimensional data", "Prevents overfitting"],
                limitations: ["Can be unstable with correlated features", "May select only one from correlated group"]
            },
            {
                name: "SVM (Support Vector Machine) Regression",
                description: "Uses support vector machines for regression by finding a hyperplane that best fits the data within a margin.",
                useCases: [
                    "Stock market price prediction with complex patterns",
                    "Energy consumption forecasting",
                    "Time series prediction with non-linear relationships"
                ],
                strengths: ["Handles non-linear relationships", "Robust to outliers", "Works in high dimensions"],
                limitations: ["Computationally expensive", "Requires careful parameter tuning", "Less interpretable"]
            },
            {
                name: "Decision Tree Regression",
                description: "Creates a tree-like model of decisions to predict continuous values by splitting data recursively.",
                useCases: [
                    "Customer lifetime value prediction",
                    "Real estate appraisal with mixed feature types",
                    "Medical dosage optimization based on patient characteristics"
                ],
                strengths: ["Handles non-linear relationships", "No feature scaling needed", "Easy to visualize"],
                limitations: ["Prone to overfitting", "Unstable with small data changes", "Biased with imbalanced data"]
            },
            {
                name: "Random Forest Regression",
                description: "Ensemble of decision trees that averages predictions to improve accuracy and reduce overfitting.",
                useCases: [
                    "Demand forecasting for retail inventory management",
                    "Insurance claim amount prediction",
                    "Agricultural yield prediction based on weather and soil data"
                ],
                strengths: ["High accuracy", "Handles missing values", "Reduces overfitting", "Feature importance"],
                limitations: ["Less interpretable", "Computationally intensive", "Large model size"]
            },
            {
                name: "Neural Networks",
                description: "Multi-layer networks that learn complex non-linear relationships through backpropagation.",
                useCases: [
                    "Complex financial modeling and derivatives pricing",
                    "Weather pattern prediction",
                    "Manufacturing quality prediction with sensor data"
                ],
                strengths: ["Learns complex patterns", "Handles large datasets", "Flexible architecture"],
                limitations: ["Requires large data", "Black box model", "Computationally expensive", "Needs tuning"]
            },
            {
                name: "GBM (Gradient Boosting Machine)",
                description: "Sequential ensemble method that builds trees to correct errors of previous models.",
                useCases: [
                    "Credit scoring and risk assessment",
                    "Click-through rate prediction in advertising",
                    "Customer churn prediction with complex patterns"
                ],
                strengths: ["State-of-the-art accuracy", "Handles mixed data types", "Feature importance"],
                limitations: ["Prone to overfitting", "Sensitive to parameters", "Longer training time"]
            },
            {
                name: "GLM (Generalized Linear Model)",
                description: "Extension of linear regression allowing for non-normal distributions and link functions.",
                useCases: [
                    "Insurance claim frequency modeling (Poisson distribution)",
                    "Medical outcome prediction with binary or count data",
                    "Marketing response modeling"
                ],
                strengths: ["Flexible distributions", "Interpretable coefficients", "Handles various data types"],
                limitations: ["Assumes distribution family", "Limited to exponential family", "Less flexible than ML"]
            },
            {
                name: "K-Nearest Neighbors",
                description: "Non-parametric method that predicts based on average of K nearest training examples.",
                useCases: [
                    "Real-time price prediction for similar products",
                    "Collaborative filtering for rating prediction",
                    "Local weather forecasting based on nearby stations"
                ],
                strengths: ["Simple and intuitive", "No training phase", "Adapts to local patterns"],
                limitations: ["Slow prediction", "Sensitive to feature scaling", "Curse of dimensionality"]
            },
            {
                name: "Stepwise Regression",
                description: "Automated feature selection method that adds or removes predictors based on statistical criteria.",
                useCases: [
                    "Building parsimonious models in social sciences",
                    "Identifying key factors in marketing mix modeling",
                    "Clinical research for determining significant predictors"
                ],
                strengths: ["Automatic variable selection", "Reduces model complexity", "Easy to implement"],
                limitations: ["Can miss interactions", "Multiple testing issues", "Not stable with collinearity"]
            },
            {
                name: "Quantile Regression",
                description: "Estimates conditional quantiles rather than conditional mean, useful for understanding distribution.",
                useCases: [
                    "Risk assessment in finance (VaR calculation)",
                    "Growth chart development in pediatrics",
                    "Extreme weather prediction (tail events)"
                ],
                strengths: ["Robust to outliers", "Provides full distribution", "Flexible modeling"],
                limitations: ["More complex interpretation", "Computationally intensive", "Requires larger samples"]
            }
        ]
    },
    classification: {
        title: "Classification",
        icon: "fa-tags",
        color: "purple",
        description: "For sorting things into categories - spam or not spam, cat or dog, etc.",
        algorithms: [
            {
                name: "Logistic Regression",
                description: "Despite its name, this is perfect for yes/no decisions. It calculates the probability of something belonging to one category or another.",
                useCases: [
                    "Email spam detection (spam vs. not spam)",
                    "Disease diagnosis (diseased vs. healthy)",
                    "Customer churn prediction (will churn vs. won't churn)"
                ],
                strengths: ["Probabilistic output", "Fast and efficient", "Interpretable coefficients"],
                limitations: ["Assumes linearity", "Binary or ordinal outcomes", "Sensitive to outliers"]
            },
            {
                name: "Random Forest Classification",
                description: "Ensemble of decision trees using voting mechanism for robust classification.",
                useCases: [
                    "Credit card fraud detection",
                    "Medical diagnosis with multiple symptoms",
                    "Customer segmentation for targeted marketing"
                ],
                strengths: ["High accuracy", "Handles imbalanced data", "Feature importance", "Robust"],
                limitations: ["Black box model", "Large memory footprint", "Slower prediction"]
            },
            {
                name: "Naive Bayes",
                description: "Probabilistic classifier based on Bayes' theorem with independence assumptions.",
                useCases: [
                    "Text classification and sentiment analysis",
                    "Real-time email filtering",
                    "Document categorization for content management"
                ],
                strengths: ["Fast training and prediction", "Works with small data", "Handles high dimensions"],
                limitations: ["Independence assumption", "Requires good probability estimates", "Zero frequency problem"]
            },
            {
                name: "SVM (Support Vector Machine)",
                description: "Finds optimal hyperplane that maximally separates classes in feature space.",
                useCases: [
                    "Image classification and face recognition",
                    "Protein structure prediction in bioinformatics",
                    "Handwritten digit recognition"
                ],
                strengths: ["Effective in high dimensions", "Memory efficient", "Versatile kernels"],
                limitations: ["Slow with large datasets", "Sensitive to parameters", "No probability estimates"]
            },
            {
                name: "Decision Tree Classification",
                description: "Tree structure where internal nodes test features and leaves represent class labels.",
                useCases: [
                    "Loan approval decision systems",
                    "Medical diagnosis with clear decision paths",
                    "Customer risk profiling"
                ],
                strengths: ["Easy to interpret", "Handles mixed data", "No preprocessing needed"],
                limitations: ["Overfitting tendency", "Unstable", "Biased with imbalanced classes"]
            },
            {
                name: "Neural Networks",
                description: "Deep learning models with multiple layers for complex pattern recognition.",
                useCases: [
                    "Image recognition and computer vision",
                    "Speech recognition and NLP tasks",
                    "Complex pattern classification in medical imaging"
                ],
                strengths: ["Learns complex features", "State-of-the-art performance", "Handles large data"],
                limitations: ["Requires large data", "Computationally expensive", "Black box", "Overfitting risk"]
            },
            {
                name: "GBM (Gradient Boosting)",
                description: "Sequential ensemble building strong classifier from weak learners.",
                useCases: [
                    "Kaggle competitions and complex classification tasks",
                    "Click prediction in online advertising",
                    "Risk assessment in insurance"
                ],
                strengths: ["Excellent accuracy", "Handles mixed types", "Feature engineering automatic"],
                limitations: ["Overfitting risk", "Requires tuning", "Longer training", "Less interpretable"]
            },
            {
                name: "K-Nearest Neighbors",
                description: "Assigns class based on majority vote of K nearest neighbors.",
                useCases: [
                    "Recommendation systems (similar user classification)",
                    "Pattern recognition in small datasets",
                    "Anomaly detection in network security"
                ],
                strengths: ["Simple and intuitive", "No training required", "Naturally handles multi-class"],
                limitations: ["Computationally expensive prediction", "Sensitive to irrelevant features", "Needs feature scaling"]
            },
            {
                name: "Multiclass Naive Bayes",
                description: "Extension of Naive Bayes for multiple class labels using probabilistic framework.",
                useCases: [
                    "News article categorization (sports, politics, tech, etc.)",
                    "Product categorization in e-commerce",
                    "Multi-label sentiment classification"
                ],
                strengths: ["Fast and scalable", "Works with small training data", "Handles many classes"],
                limitations: ["Independence assumption", "May underperform with correlated features"]
            }
        ]
    },
    clustering: {
        title: "Clustering",
        icon: "fa-object-group",
        color: "pink",
        description: "Finding natural groups in your data without predefined labels",
        algorithms: [
            {
                name: "K-Means",
                description: "Groups your data into K clusters by putting similar items together. You tell it how many groups you want, and it figures out the rest.",
                useCases: [
                    "Customer segmentation for marketing campaigns",
                    "Image compression by color quantization",
                    "Document clustering for topic discovery"
                ],
                strengths: ["Fast and scalable", "Simple to implement", "Works well with spherical clusters"],
                limitations: ["Requires K specification", "Sensitive to outliers", "Assumes spherical clusters"]
            },
            {
                name: "Hierarchical Clustering",
                description: "Builds tree of clusters using agglomerative or divisive approaches.",
                useCases: [
                    "Taxonomy creation in biology",
                    "Social network analysis for community detection",
                    "Gene expression analysis in genomics"
                ],
                strengths: ["No K needed upfront", "Dendrogram visualization", "Flexible distance metrics"],
                limitations: ["Computationally expensive", "Sensitive to noise", "Cannot undo merges"]
            },
            {
                name: "DBSCAN",
                description: "Density-based clustering that finds clusters of arbitrary shape and identifies outliers.",
                useCases: [
                    "Anomaly detection in network traffic",
                    "Geographic data clustering (finding city centers)",
                    "Identifying patterns in spatial data"
                ],
                strengths: ["Finds arbitrary shapes", "Robust to outliers", "No K needed"],
                limitations: ["Sensitive to parameters", "Struggles with varying densities", "Not deterministic"]
            },
            {
                name: "Gaussian Mixture Model",
                description: "Probabilistic model assuming data comes from mixture of Gaussian distributions.",
                useCases: [
                    "Image segmentation in computer vision",
                    "Speaker identification in audio processing",
                    "Financial market regime detection"
                ],
                strengths: ["Soft clustering", "Probabilistic framework", "Flexible cluster shapes"],
                limitations: ["Assumes Gaussian distributions", "Sensitive to initialization", "Can overfit"]
            },
            {
                name: "Affinity Propagation",
                description: "Message-passing algorithm that automatically determines number of clusters.",
                useCases: [
                    "Face recognition for identifying exemplar faces",
                    "Gene clustering in bioinformatics",
                    "Image categorization"
                ],
                strengths: ["Automatic K selection", "Identifies exemplars", "No random initialization"],
                limitations: ["Computationally expensive", "Memory intensive", "Sensitive to preferences"]
            },
            {
                name: "Spectral Clustering",
                description: "Uses eigenvalues of similarity matrix for dimensionality reduction before clustering.",
                useCases: [
                    "Image segmentation with complex boundaries",
                    "Community detection in social networks",
                    "Clustering of non-convex shapes"
                ],
                strengths: ["Handles non-convex clusters", "Works with similarity graphs", "Flexible"],
                limitations: ["Computationally expensive", "Requires K specification", "Sensitive to parameters"]
            }
        ]
    },
    computerVision: {
        title: "Computer Vision",
        icon: "fa-eye",
        color: "green",
        description: "Teaching computers to see and understand images and videos",
        algorithms: [
            {
                name: "CNN (Convolutional Neural Network)",
                description: "The workhorse of computer vision. These networks are specifically designed to process images by learning features like edges, shapes, and patterns automatically.",
                useCases: [
                    "Image classification (identifying objects in photos)",
                    "Medical image analysis (detecting tumors in X-rays)",
                    "Facial recognition systems"
                ],
                strengths: ["Automatic feature extraction", "Translation invariance", "Hierarchical learning"],
                limitations: ["Requires large datasets", "Computationally intensive", "Black box model"]
            },
            {
                name: "Fast R-CNN",
                description: "Region-based CNN for object detection with improved speed over R-CNN.",
                useCases: [
                    "Autonomous vehicle object detection",
                    "Retail inventory management (counting products)",
                    "Security surveillance for threat detection"
                ],
                strengths: ["Fast inference", "Accurate bounding boxes", "End-to-end training"],
                limitations: ["Still slower than YOLO", "Complex architecture", "Requires region proposals"]
            },
            {
                name: "GANs (Generative Adversarial Networks)",
                description: "Two neural networks (generator and discriminator) competing to create realistic synthetic data.",
                useCases: [
                    "Photo-realistic image generation",
                    "Data augmentation for training",
                    "Art creation and style transfer",
                    "Face aging and de-aging effects"
                ],
                strengths: ["Generates realistic data", "Unsupervised learning", "Creative applications"],
                limitations: ["Training instability", "Mode collapse", "Requires careful tuning"]
            },
            {
                name: "YOLO (You Only Look Once)",
                description: "Real-time object detection system that predicts bounding boxes and class probabilities.",
                useCases: [
                    "Real-time video surveillance",
                    "Autonomous driving perception systems",
                    "Sports analytics (tracking players and ball)"
                ],
                strengths: ["Extremely fast", "Real-time detection", "Good generalization"],
                limitations: ["Less accurate on small objects", "Struggles with close objects", "Fixed grid limitations"]
            },
            {
                name: "ImageNet",
                description: "Large-scale dataset and benchmark for image classification with pre-trained models.",
                useCases: [
                    "Transfer learning for custom image tasks",
                    "Feature extraction for downstream tasks",
                    "Benchmark for new model architectures"
                ],
                strengths: ["Comprehensive dataset", "Pre-trained models", "Industry standard"],
                limitations: ["Dataset bias", "Limited to 1000 classes", "Static dataset"]
            },
            {
                name: "U-Net",
                description: "CNN architecture designed for biomedical image segmentation with skip connections.",
                useCases: [
                    "Medical image segmentation (organs, tumors)",
                    "Satellite image analysis (land use classification)",
                    "Cell detection in microscopy images"
                ],
                strengths: ["Works with small datasets", "Precise segmentation", "Fast training"],
                limitations: ["Specific to segmentation", "Memory intensive", "Fixed input size"]
            },
            {
                name: "AlexNet",
                description: "Pioneering deep CNN that won ImageNet 2012, popularizing deep learning for vision.",
                useCases: [
                    "Historical benchmark for image classification",
                    "Educational purposes for understanding CNNs",
                    "Transfer learning baseline"
                ],
                strengths: ["Breakthrough architecture", "Proved deep learning viability", "Easy to understand"],
                limitations: ["Outdated by modern standards", "Large model size", "Less efficient"]
            },
            {
                name: "VGG",
                description: "Very deep CNN with small 3x3 filters, known for simplicity and depth.",
                useCases: [
                    "Feature extraction for style transfer",
                    "Image classification baseline",
                    "Visual similarity search"
                ],
                strengths: ["Simple architecture", "Good feature representations", "Transfer learning"],
                limitations: ["Very large model", "Slow inference", "Memory intensive"]
            },
            {
                name: "ResNet",
                description: "Revolutionary architecture with residual connections enabling training of very deep networks.",
                useCases: [
                    "State-of-the-art image classification",
                    "Object detection backbone",
                    "Medical imaging analysis"
                ],
                strengths: ["Trains very deep networks", "Avoids vanishing gradients", "Excellent performance"],
                limitations: ["Complex architecture", "Computationally expensive", "Large model size"]
            }
        ]
    },
    recommender: {
        title: "Recommender Systems",
        icon: "fa-star",
        color: "amber",
        description: "Suggesting things people might like based on their preferences and behavior",
        algorithms: [
            {
                name: "Collaborative Filtering",
                description: "Recommends items by finding users with similar tastes. If you and I both liked the same movies, and I watched something you haven't, it'll suggest that to you.",
                useCases: [
                    "Netflix movie recommendations",
                    "Amazon product suggestions",
                    "Spotify music recommendations"
                ],
                strengths: ["No domain knowledge needed", "Discovers unexpected patterns", "Improves with data"],
                limitations: ["Cold start problem", "Sparsity issues", "Scalability challenges"]
            },
            {
                name: "Alternating Least Squares",
                description: "Matrix factorization technique for collaborative filtering that scales to large datasets.",
                useCases: [
                    "Large-scale recommendation systems (Spotify, Netflix)",
                    "Implicit feedback scenarios (views, clicks)",
                    "Distributed recommendation engines"
                ],
                strengths: ["Scalable to big data", "Handles implicit feedback", "Parallelizable"],
                limitations: ["Cold start problem", "Requires tuning", "Linear complexity"]
            },
            {
                name: "Content Filtering",
                description: "Recommends items similar to what user liked based on item features and user profile.",
                useCases: [
                    "News article recommendations based on reading history",
                    "Job recommendations matching skills and experience",
                    "Product recommendations based on specifications"
                ],
                strengths: ["No cold start for items", "Transparent recommendations", "User independence"],
                limitations: ["Limited discovery", "Requires feature engineering", "Over-specialization"]
            },
            {
                name: "Association Rule Mining",
                description: "Discovers relationships between items using market basket analysis (if-then rules).",
                useCases: [
                    "Market basket analysis (beer and diapers)",
                    "Cross-selling recommendations in e-commerce",
                    "Web usage mining for page recommendations"
                ],
                strengths: ["Interpretable rules", "Discovers patterns", "No training needed"],
                limitations: ["Combinatorial explosion", "Requires frequent patterns", "Static rules"]
            },
            {
                name: "RNN (Recurrent Neural Network)",
                description: "Sequential model that captures temporal patterns in user behavior for recommendations.",
                useCases: [
                    "Session-based recommendations (next item prediction)",
                    "Sequential music playlist generation",
                    "Video streaming recommendations"
                ],
                strengths: ["Captures sequences", "Temporal patterns", "Context-aware"],
                limitations: ["Computationally expensive", "Requires large data", "Training complexity"]
            },
            {
                name: "Two-Tower Model",
                description: "Neural architecture with separate towers for users and items, computing similarity in embedding space.",
                useCases: [
                    "YouTube video recommendations at scale",
                    "Large-scale retrieval systems",
                    "Real-time personalization"
                ],
                strengths: ["Scalable inference", "Flexible architecture", "Handles large catalogs"],
                limitations: ["Requires large data", "Complex training", "Embedding quality critical"]
            }
        ]
    },
    forecasting: {
        title: "Forecasting",
        icon: "fa-chart-area",
        color: "red",
        description: "Predicting what's likely to happen in the future based on past trends",
        algorithms: [
            {
                name: "ARIMA",
                description: "AutoRegressive Integrated Moving Average model for time series forecasting.",
                useCases: [
                    "Stock price prediction",
                    "Sales forecasting for inventory planning",
                    "Economic indicator prediction (GDP, inflation)"
                ],
                strengths: ["Well-established theory", "Handles trends", "Interpretable parameters"],
                limitations: ["Assumes stationarity", "Manual parameter selection", "Linear relationships"]
            },
            {
                name: "SARIMA",
                description: "Seasonal ARIMA that extends ARIMA to handle seasonal patterns in data.",
                useCases: [
                    "Retail sales with holiday seasonality",
                    "Energy consumption forecasting",
                    "Tourism demand prediction"
                ],
                strengths: ["Handles seasonality", "Flexible modeling", "Proven track record"],
                limitations: ["Complex parameter tuning", "Assumes linear seasonality", "Computational cost"]
            },
            {
                name: "SARIMAX",
                description: "SARIMA with exogenous variables allowing external factors in forecasting.",
                useCases: [
                    "Sales forecasting with marketing spend as input",
                    "Weather-dependent demand prediction",
                    "Traffic flow prediction with event data"
                ],
                strengths: ["Incorporates external factors", "Flexible framework", "Improved accuracy"],
                limitations: ["Requires relevant exogenous variables", "Complex specification", "Overfitting risk"]
            },
            {
                name: "Exponential Smoothing",
                description: "Weighted average of past observations with exponentially decreasing weights.",
                useCases: [
                    "Short-term demand forecasting",
                    "Inventory management",
                    "Real-time monitoring and alerting"
                ],
                strengths: ["Simple and fast", "Adaptive to recent data", "Good for short-term"],
                limitations: ["Limited to simple patterns", "No confidence intervals", "Short-term focus"]
            },
            {
                name: "Deep AR",
                description: "Deep learning autoregressive model using RNNs for probabilistic forecasting.",
                useCases: [
                    "Multi-step ahead forecasting",
                    "Demand forecasting for thousands of products",
                    "Energy load forecasting"
                ],
                strengths: ["Probabilistic forecasts", "Handles multiple series", "Captures complex patterns"],
                limitations: ["Requires large data", "Computationally intensive", "Black box"]
            },
            {
                name: "BayesDLM",
                description: "Bayesian Dynamic Linear Model providing probabilistic time series forecasting.",
                useCases: [
                    "Financial risk modeling",
                    "Epidemiological forecasting",
                    "Climate modeling with uncertainty"
                ],
                strengths: ["Uncertainty quantification", "Flexible framework", "Online learning"],
                limitations: ["Computationally intensive", "Requires prior specification", "Complex implementation"]
            },
            {
                name: "N-Beats",
                description: "Neural basis expansion analysis for time series, pure deep learning approach.",
                useCases: [
                    "M4 competition winning model",
                    "General purpose forecasting",
                    "Complex seasonal patterns"
                ],
                strengths: ["State-of-the-art accuracy", "Interpretable components", "No feature engineering"],
                limitations: ["Requires large data", "Computationally expensive", "Less interpretable"]
            }
        ]
    },
    optimization: {
        title: "Optimization",
        icon: "fa-sliders-h",
        color: "cyan",
        description: "Finding the best solution when you have lots of options and constraints",
        algorithms: [
            {
                name: "Stochastic Search",
                description: "Random search methods that explore solution space probabilistically.",
                useCases: [
                    "Hyperparameter tuning in machine learning",
                    "Portfolio optimization with uncertainty",
                    "Complex engineering design problems"
                ],
                strengths: ["Handles non-convex problems", "No gradient needed", "Explores broadly"],
                limitations: ["Slow convergence", "No optimality guarantee", "Many evaluations needed"]
            },
            {
                name: "Genetic Algorithm",
                description: "Evolutionary algorithm inspired by natural selection using crossover and mutation.",
                useCases: [
                    "Feature selection in machine learning",
                    "Vehicle routing and scheduling",
                    "Neural architecture search"
                ],
                strengths: ["Handles complex landscapes", "Parallel evaluation", "Flexible encoding"],
                limitations: ["Computationally expensive", "Many parameters", "No convergence guarantee"]
            },
            {
                name: "Gradient Search",
                description: "Optimization using gradient information to find local minima iteratively.",
                useCases: [
                    "Training neural networks (backpropagation)",
                    "Logistic regression optimization",
                    "Maximum likelihood estimation"
                ],
                strengths: ["Fast convergence", "Efficient", "Well-understood theory"],
                limitations: ["Requires differentiability", "Can stuck in local minima", "Sensitive to learning rate"]
            },
            {
                name: "Linear Programming",
                description: "Optimization of linear objective function subject to linear constraints.",
                useCases: [
                    "Supply chain optimization",
                    "Production planning and scheduling",
                    "Resource allocation in projects"
                ],
                strengths: ["Guaranteed global optimum", "Efficient algorithms", "Well-established"],
                limitations: ["Limited to linear problems", "Infeasible solutions possible", "Assumes certainty"]
            },
            {
                name: "Integer Programming",
                description: "Linear programming where some or all variables must be integers.",
                useCases: [
                    "Facility location problems",
                    "Job scheduling with discrete decisions",
                    "Network design and routing"
                ],
                strengths: ["Handles discrete decisions", "Exact solutions", "Flexible modeling"],
                limitations: ["NP-hard complexity", "Computationally expensive", "May not scale"]
            },
            {
                name: "Multi-Armed Bandit",
                description: "Sequential decision problem balancing exploration and exploitation.",
                useCases: [
                    "A/B testing and website optimization",
                    "Clinical trial design",
                    "Online advertising campaign optimization"
                ],
                strengths: ["Online learning", "Balances exploration-exploitation", "Regret bounds"],
                limitations: ["Assumes stationarity", "Limited to simple settings", "Contextual variants complex"]
            }
        ]
    },
    nlp: {
        title: "NLP / LLM",
        icon: "fa-language",
        color: "indigo",
        description: "Helping computers understand, process, and generate human language",
        algorithms: [
            {
                name: "RNN (Recurrent Neural Network)",
                description: "Neural network with loops allowing information persistence for sequential data.",
                useCases: [
                    "Language modeling and text generation",
                    "Machine translation",
                    "Speech recognition"
                ],
                strengths: ["Handles variable length sequences", "Shares parameters across time", "Memory of past"],
                limitations: ["Vanishing gradient problem", "Slow training", "Limited long-term memory"]
            },
            {
                name: "LSTM (Long Short-Term Memory)",
                description: "Advanced RNN with gates to better capture long-term dependencies.",
                useCases: [
                    "Sentiment analysis of long documents",
                    "Named entity recognition",
                    "Video captioning"
                ],
                strengths: ["Solves vanishing gradient", "Long-term dependencies", "Proven effectiveness"],
                limitations: ["Computationally expensive", "Sequential processing", "Still limited context"]
            },
            {
                name: "BERT",
                description: "Bidirectional transformer pre-trained on large text corpus for understanding context.",
                useCases: [
                    "Question answering systems",
                    "Semantic search and information retrieval",
                    "Text classification and NER"
                ],
                strengths: ["Bidirectional context", "Transfer learning", "State-of-the-art performance"],
                limitations: ["Large model size", "Slow inference", "Fixed input length"]
            },
            {
                name: "GPT (Generative Pre-trained Transformer)",
                description: "Autoregressive language model trained to predict next token, enabling text generation.",
                useCases: [
                    "Content creation and copywriting",
                    "Code generation and completion",
                    "Conversational AI and chatbots"
                ],
                strengths: ["Excellent generation", "Few-shot learning", "Versatile applications"],
                limitations: ["Unidirectional", "Hallucination issues", "Large computational requirements"]
            },
            {
                name: "Word2Vec",
                description: "Neural network model creating dense word embeddings capturing semantic relationships.",
                useCases: [
                    "Word similarity and analogy tasks",
                    "Feature extraction for text classification",
                    "Semantic search applications"
                ],
                strengths: ["Captures semantics", "Efficient training", "Pre-trained models available"],
                limitations: ["Fixed vocabulary", "No context awareness", "Polysemy issues"]
            },
            {
                name: "LaMDA",
                description: "Google's Language Model for Dialogue Applications designed for conversational AI.",
                useCases: [
                    "Open-domain conversational agents",
                    "Customer service chatbots",
                    "Interactive assistants"
                ],
                strengths: ["Natural conversations", "Context awareness", "Safety filtering"],
                limitations: ["Proprietary", "Large scale requirements", "Potential biases"]
            },
            {
                name: "Llama",
                description: "Meta's open-source large language model family optimized for efficiency.",
                useCases: [
                    "Research and academic applications",
                    "Custom fine-tuning for specific domains",
                    "On-premise AI deployments"
                ],
                strengths: ["Open source", "Efficient architecture", "Good performance"],
                limitations: ["Still requires significant resources", "License restrictions", "Training data biases"]
            },
            {
                name: "StableLM",
                description: "Stability AI's open-source language model series for various NLP tasks.",
                useCases: [
                    "Open-source AI applications",
                    "Educational and research purposes",
                    "Custom model development"
                ],
                strengths: ["Open and accessible", "Multiple sizes", "Commercial friendly"],
                limitations: ["Smaller than proprietary models", "Performance trade-offs", "Limited support"]
            }
        ]
    }
};

// Category color mapping
const colorMap = {
    blue: { bg: 'bg-blue-50', text: 'text-blue-600', border: 'border-blue-500', hover: 'hover:bg-blue-100' },
    purple: { bg: 'bg-purple-50', text: 'text-purple-600', border: 'border-purple-500', hover: 'hover:bg-purple-100' },
    pink: { bg: 'bg-pink-50', text: 'text-pink-600', border: 'border-pink-500', hover: 'hover:bg-pink-100' },
    green: { bg: 'bg-green-50', text: 'text-green-600', border: 'border-green-500', hover: 'hover:bg-green-100' },
    amber: { bg: 'bg-amber-50', text: 'text-amber-600', border: 'border-amber-500', hover: 'hover:bg-amber-100' },
    red: { bg: 'bg-red-50', text: 'text-red-600', border: 'border-red-500', hover: 'hover:bg-red-100' },
    cyan: { bg: 'bg-cyan-50', text: 'text-cyan-600', border: 'border-cyan-500', hover: 'hover:bg-cyan-100' },
    indigo: { bg: 'bg-indigo-50', text: 'text-indigo-600', border: 'border-indigo-500', hover: 'hover:bg-indigo-100' }
};

// Render categories
function renderCategories(data = mlData) {
    const container = document.getElementById('categoriesContainer');
    container.innerHTML = '';
    
    Object.keys(data).forEach(key => {
        const category = data[key];
        const colors = colorMap[category.color];
        
        const card = document.createElement('div');
        card.className = `category-card ${key} bg-white rounded-xl shadow-lg p-6 cursor-pointer hover:shadow-2xl`;
        
        card.innerHTML = `
            <div class="flex items-start mb-4">
                <div class="w-12 h-12 ${colors.bg} rounded-lg flex items-center justify-center mr-4">
                    <i class="fas ${category.icon} ${colors.text} text-xl"></i>
                </div>
                <div class="flex-1">
                    <h3 class="text-2xl font-bold text-gray-800 mb-2">${category.title}</h3>
                    <p class="text-gray-600 text-sm">${category.description}</p>
                </div>
            </div>
            
            <div class="mb-4">
                <p class="text-sm font-semibold text-gray-700 mb-2">
                    <i class="fas fa-code mr-2"></i>${category.algorithms.length} Algorithms
                </p>
            </div>
            
            <div class="flex flex-wrap gap-2 mb-4">
                ${category.algorithms.slice(0, 6).map(algo => `
                    <span class="algorithm-tag ${colors.bg} ${colors.text} px-3 py-1 rounded-full text-xs font-medium cursor-pointer ${colors.hover}"
                          onclick="showAlgorithmDetails('${key}', '${algo.name}')">
                        ${algo.name}
                    </span>
                `).join('')}
                ${category.algorithms.length > 6 ? `
                    <span class="${colors.bg} ${colors.text} px-3 py-1 rounded-full text-xs font-medium">
                        +${category.algorithms.length - 6} more
                    </span>
                ` : ''}
            </div>
            
            <button onclick="showCategoryDetails('${key}')" 
                    class="w-full ${colors.bg} ${colors.text} py-2 rounded-lg font-semibold ${colors.hover} transition">
                View All Algorithms <i class="fas fa-arrow-right ml-2"></i>
            </button>
        `;
        
        container.appendChild(card);
    });
}

// Show category details
function showCategoryDetails(categoryKey) {
    const category = mlData[categoryKey];
    const colors = colorMap[category.color];
    const modal = document.getElementById('algorithmModal');
    
    document.getElementById('modalTitle').textContent = category.title;
    document.getElementById('modalCategory').textContent = `${category.algorithms.length} Algorithms`;
    
    const content = `
        <div class="mb-6">
            <p class="text-gray-700 text-lg">${category.description}</p>
        </div>
        
        <div class="space-y-4">
            ${category.algorithms.map(algo => `
                <div class="${colors.bg} rounded-lg p-4 cursor-pointer hover:shadow-md transition"
                     onclick="showAlgorithmDetails('${categoryKey}', '${algo.name}')">
                    <h4 class="font-bold text-lg ${colors.text} mb-2">
                        <i class="fas fa-chevron-right mr-2"></i>${algo.name}
                    </h4>
                    <p class="text-gray-600 text-sm">${algo.description}</p>
                    <div class="mt-2">
                        <span class="text-xs ${colors.text} font-semibold">
                            Click to see use cases and details →
                        </span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    document.getElementById('modalContent').innerHTML = content;
    modal.style.display = 'block';
}

// Show algorithm details
function showAlgorithmDetails(categoryKey, algorithmName) {
    const category = mlData[categoryKey];
    const algorithm = category.algorithms.find(a => a.name === algorithmName);
    const colors = colorMap[category.color];
    const modal = document.getElementById('algorithmModal');
    
    document.getElementById('modalTitle').textContent = algorithm.name;
    document.getElementById('modalCategory').textContent = category.title;
    
    const content = `
        <div class="space-y-6">
            <div>
                <h3 class="text-xl font-bold ${colors.text} mb-3">
                    <i class="fas fa-info-circle mr-2"></i>Description
                </h3>
                <p class="text-gray-700 leading-relaxed">${algorithm.description}</p>
            </div>
            
            <div class="${colors.bg} rounded-lg p-5">
                <h3 class="text-xl font-bold ${colors.text} mb-3">
                    <i class="fas fa-lightbulb mr-2"></i>Real-World Use Cases
                </h3>
                <ul class="space-y-2">
                    ${algorithm.useCases.map(useCase => `
                        <li class="flex items-start">
                            <i class="fas fa-check-circle ${colors.text} mt-1 mr-3"></i>
                            <span class="text-gray-700">${useCase}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
            
            <div class="grid md:grid-cols-2 gap-4">
                <div class="bg-green-50 rounded-lg p-4">
                    <h4 class="font-bold text-green-700 mb-3">
                        <i class="fas fa-thumbs-up mr-2"></i>Strengths
                    </h4>
                    <ul class="space-y-1">
                        ${algorithm.strengths.map(strength => `
                            <li class="text-sm text-gray-700">
                                <i class="fas fa-plus text-green-600 mr-2"></i>${strength}
                            </li>
                        `).join('')}
                    </ul>
                </div>
                
                <div class="bg-red-50 rounded-lg p-4">
                    <h4 class="font-bold text-red-700 mb-3">
                        <i class="fas fa-exclamation-triangle mr-2"></i>Limitations
                    </h4>
                    <ul class="space-y-1">
                        ${algorithm.limitations.map(limitation => `
                            <li class="text-sm text-gray-700">
                                <i class="fas fa-minus text-red-600 mr-2"></i>${limitation}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            </div>
            
            <div class="text-center pt-4">
                <button onclick="showCategoryDetails('${categoryKey}')" 
                        class="${colors.bg} ${colors.text} px-6 py-2 rounded-lg font-semibold hover:shadow-md transition">
                    <i class="fas fa-arrow-left mr-2"></i>Back to ${category.title}
                </button>
            </div>
        </div>
    `;
    
    document.getElementById('modalContent').innerHTML = content;
    modal.style.display = 'block';
}

// Search functionality
document.getElementById('searchInput').addEventListener('input', function(e) {
    const searchTerm = e.target.value.toLowerCase();
    
    if (searchTerm === '') {
        renderCategories();
        return;
    }
    
    const filteredData = {};
    
    Object.keys(mlData).forEach(key => {
        const category = mlData[key];
        const matchingAlgorithms = category.algorithms.filter(algo => {
            return algo.name.toLowerCase().includes(searchTerm) ||
                   algo.description.toLowerCase().includes(searchTerm) ||
                   algo.useCases.some(uc => uc.toLowerCase().includes(searchTerm));
        });
        
        if (matchingAlgorithms.length > 0 || 
            category.title.toLowerCase().includes(searchTerm) ||
            category.description.toLowerCase().includes(searchTerm)) {
            filteredData[key] = {
                ...category,
                algorithms: matchingAlgorithms.length > 0 ? matchingAlgorithms : category.algorithms
            };
        }
    });
    
    renderCategories(filteredData);
});

// Modal close functionality
document.querySelector('.close').onclick = function() {
    document.getElementById('algorithmModal').style.display = 'none';
}

window.onclick = function(event) {
    const modal = document.getElementById('algorithmModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    renderCategories();
});
