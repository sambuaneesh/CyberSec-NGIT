"use strict";(self.webpackChunkdocumentation=self.webpackChunkdocumentation||[]).push([[70],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>g});var a=r(7294);function s(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function n(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,a)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?n(Object(r),!0).forEach((function(t){s(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):n(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function o(e,t){if(null==e)return{};var r,a,s=function(e,t){if(null==e)return{};var r,a,s={},n=Object.keys(e);for(a=0;a<n.length;a++)r=n[a],t.indexOf(r)>=0||(s[r]=e[r]);return s}(e,t);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);for(a=0;a<n.length;a++)r=n[a],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(s[r]=e[r])}return s}var l=a.createContext({}),c=function(e){var t=a.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},u=function(e){var t=c(e.components);return a.createElement(l.Provider,{value:t},e.children)},p="mdxType",h={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var r=e.components,s=e.mdxType,n=e.originalType,l=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),p=c(r),d=s,g=p["".concat(l,".").concat(d)]||p[d]||h[d]||n;return r?a.createElement(g,i(i({ref:t},u),{},{components:r})):a.createElement(g,i({ref:t},u))}));function g(e,t){var r=arguments,s=t&&t.mdxType;if("string"==typeof e||s){var n=r.length,i=new Array(n);i[0]=d;var o={};for(var l in t)hasOwnProperty.call(t,l)&&(o[l]=t[l]);o.originalType=e,o[p]="string"==typeof e?e:s,i[1]=o;for(var c=2;c<n;c++)i[c]=r[c];return a.createElement.apply(null,i)}return a.createElement.apply(null,r)}d.displayName="MDXCreateElement"},9198:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>l,contentTitle:()=>i,default:()=>h,frontMatter:()=>n,metadata:()=>o,toc:()=>c});var a=r(7462),s=(r(7294),r(3905));const n={slug:"regression",title:"What are Regressive tasks?",authors:"aneesh",tags:["regression","classification","ml","algo"]},i=void 0,o={permalink:"/CyberSec-NGIT/blog/regression",editUrl:"https://github.com/stealthspectre/CyberSec-NGIT/blog/regression.md",source:"@site/blog/regression.md",title:"What are Regressive tasks?",description:"So what's this Regression?",date:"2023-03-29T10:54:53.139Z",formattedDate:"March 29, 2023",tags:[{label:"regression",permalink:"/CyberSec-NGIT/blog/tags/regression"},{label:"classification",permalink:"/CyberSec-NGIT/blog/tags/classification"},{label:"ml",permalink:"/CyberSec-NGIT/blog/tags/ml"},{label:"algo",permalink:"/CyberSec-NGIT/blog/tags/algo"}],readingTime:2.3,hasTruncateMarker:!1,authors:[{name:"Aneesh Sambu",title:"stealthspectre",url:"https://github.com/stealthspectre",imageURL:"https://github.com/stealthspectre.png",key:"aneesh"}],frontMatter:{slug:"regression",title:"What are Regressive tasks?",authors:"aneesh",tags:["regression","classification","ml","algo"]},prevItem:{title:"What is Gaussian?",permalink:"/CyberSec-NGIT/blog/gaussian"},nextItem:{title:"Diving into KNN",permalink:"/CyberSec-NGIT/blog/knn"}},l={authorsImageUrls:[void 0]},c=[{value:"So what&#39;s this Regression?",id:"so-whats-this-regression",level:2},{value:"What is the main difference between classification and regression?",id:"what-is-the-main-difference-between-classification-and-regression",level:2}],u={toc:c},p="wrapper";function h(e){let{components:t,...r}=e;return(0,s.kt)(p,(0,a.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("h2",{id:"so-whats-this-regression"},"So what's this Regression?"),(0,s.kt)("p",null,"In machine learning, a regression task is a type of supervised learning problem where the goal is to predict a continuous numerical value, such as a price, a temperature, or a stock price. The objective of a regression model is to learn a function that maps input features to a continuous output value."),(0,s.kt)("p",null,"Regression tasks are different from classification tasks, where the goal is to predict a categorical label, such as whether an email is spam or not. In a regression task, the output variable is a continuous value, whereas in a classification task, the output variable is a discrete value."),(0,s.kt)("p",null,"Regression models can be used for a wide range of applications, such as predicting housing prices based on features like location, square footage, and number of bedrooms, or predicting the temperature based on weather conditions like humidity, wind speed, and cloud cover."),(0,s.kt)("p",null,"There are many different types of regression models, including linear regression, polynomial regression, and decision tree regression. The choice of model depends on the specific problem and the characteristics of the data. The performance of a regression model is typically evaluated using metrics like mean squared error (MSE) or root mean squared error (RMSE), which measure the difference between the predicted values and the actual values."),(0,s.kt)("hr",null),(0,s.kt)("h2",{id:"what-is-the-main-difference-between-classification-and-regression"},"What is the main difference between classification and regression?"),(0,s.kt)("p",null,"The main difference between regression and classification tasks in machine learning is the type of output that the model is trying to predict."),(0,s.kt)("p",null,"In a regression task, the goal is to predict a continuous numerical value, such as a price, a temperature, or a stock price. The objective of a regression model is to learn a function that maps input features to a continuous output value."),(0,s.kt)("p",null,"In a classification task, on the other hand, the goal is to predict a categorical label, such as whether an email is spam or not, or whether a patient has a certain disease or not. The output variable is a discrete value, and the model is trained to classify input data into one of several predefined categories."),(0,s.kt)("p",null,"Another key difference between regression and classification tasks is the type of algorithms that are used. Regression models typically use algorithms like linear regression, polynomial regression, or decision tree regression, while classification models use algorithms like logistic regression, decision trees, or support vector machines."),(0,s.kt)("p",null,"The evaluation metrics used for regression and classification tasks are also different. For regression tasks, metrics like mean squared error (MSE) or root mean squared error (RMSE) are commonly used to measure the difference between the predicted values and the actual values. For classification tasks, metrics like accuracy, precision, recall, and F1 score are used to measure the performance of the model in correctly classifying the input data."))}h.isMDXComponent=!0}}]);