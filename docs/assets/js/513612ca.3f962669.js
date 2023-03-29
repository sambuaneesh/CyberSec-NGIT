"use strict";(self.webpackChunkdocumentation=self.webpackChunkdocumentation||[]).push([[9522],{3905:(e,t,a)=>{a.d(t,{Zo:()=>c,kt:()=>g});var n=a(7294);function o(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function r(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){o(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function s(e,t){if(null==e)return{};var a,n,o=function(e,t){if(null==e)return{};var a,n,o={},i=Object.keys(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||(o[a]=e[a]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)a=i[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(o[a]=e[a])}return o}var l=n.createContext({}),h=function(e){var t=n.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):r(r({},t),e)),a},c=function(e){var t=h(e.components);return n.createElement(l.Provider,{value:t},e.children)},u="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},p=n.forwardRef((function(e,t){var a=e.components,o=e.mdxType,i=e.originalType,l=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),u=h(a),p=o,g=u["".concat(l,".").concat(p)]||u[p]||d[p]||i;return a?n.createElement(g,r(r({ref:t},c),{},{components:a})):n.createElement(g,r({ref:t},c))}));function g(e,t){var a=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=a.length,r=new Array(i);r[0]=p;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s[u]="string"==typeof e?e:o,r[1]=s;for(var h=2;h<i;h++)r[h]=a[h];return n.createElement.apply(null,r)}return n.createElement.apply(null,a)}p.displayName="MDXCreateElement"},8644:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>l,contentTitle:()=>r,default:()=>d,frontMatter:()=>i,metadata:()=>s,toc:()=>h});var n=a(7462),o=(a(7294),a(3905));const i={slug:"knn",title:"Diving into KNN",authors:"aneesh",tags:["ml","knn","weighted knn","algo"]},r=void 0,s={permalink:"/CyberSec-NGIT/blog/knn",editUrl:"https://github.com/stealthspectre/CyberSec-NGIT/blog/knn.md",source:"@site/blog/knn.md",title:"Diving into KNN",description:"Plain Definition",date:"2023-03-29T10:45:19.848Z",formattedDate:"March 29, 2023",tags:[{label:"ml",permalink:"/CyberSec-NGIT/blog/tags/ml"},{label:"knn",permalink:"/CyberSec-NGIT/blog/tags/knn"},{label:"weighted knn",permalink:"/CyberSec-NGIT/blog/tags/weighted-knn"},{label:"algo",permalink:"/CyberSec-NGIT/blog/tags/algo"}],readingTime:4.175,hasTruncateMarker:!1,authors:[{name:"Aneesh Sambu",title:"stealthspectre",url:"https://github.com/stealthspectre",imageURL:"https://github.com/stealthspectre.png",key:"aneesh"}],frontMatter:{slug:"knn",title:"Diving into KNN",authors:"aneesh",tags:["ml","knn","weighted knn","algo"]},prevItem:{title:"What are Regressive tasks?",permalink:"/CyberSec-NGIT/blog/regression"},nextItem:{title:"What is a Parametric Algorithm?",permalink:"/CyberSec-NGIT/blog/parametric"}},l={authorsImageUrls:[void 0]},h=[{value:"Plain Definition",id:"plain-definition",level:2},{value:"Real time example",id:"real-time-example",level:2},{value:"Now what is this weighted KNN?",id:"now-what-is-this-weighted-knn",level:2}],c={toc:h},u="wrapper";function d(e){let{components:t,...a}=e;return(0,o.kt)(u,(0,n.Z)({},c,a,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"plain-definition"},"Plain Definition"),(0,o.kt)("p",null,"KNN (K-Nearest Neighbors) is a machine learning algorithm used for classification and ",(0,o.kt)("a",{parentName:"p",href:"/blog/regression"},"regression tasks"),". It is a ",(0,o.kt)("a",{parentName:"p",href:"/blog/non-parametric"},"non-parametric algorithm"),", which means that it does not make any assumptions about the underlying distribution of the data."),(0,o.kt)("p",null,"In the KNN algorithm, the input data is represented as points in a high-dimensional space, and the algorithm classifies new data points based on their proximity to the existing data points. Specifically, the algorithm calculates the distance between the new data point and each of the existing data points, and then assigns the new data point to the class that is most common among its K nearest neighbors."),(0,o.kt)("p",null,"The value of K is a hyperparameter that can be tuned to optimize the performance of the algorithm. A larger value of K will result in a smoother decision boundary, but may also lead to misclassification of data points that are close to the boundary between two classes."),(0,o.kt)("p",null,"KNN is a simple and effective algorithm that can be used for a wide range of classification and ",(0,o.kt)("a",{parentName:"p",href:"/blog/regression"},"regression tasks"),". However, it can be computationally expensive for large datasets, and may not perform well in high-dimensional spaces."),(0,o.kt)("hr",null),(0,o.kt)("h2",{id:"real-time-example"},"Real time example"),(0,o.kt)("p",null,"Imagine you are a penguin living in Antarctica, and you want to find a new place to build your igloo. You have heard that some areas are better than others, but you're not sure which ones. So, you decide to ask your penguin friends for advice."),(0,o.kt)("p",null,"You ask your friends to rate different areas on a scale of 1 to 10, based on how good they are for building an igloo. You also ask them to tell you the distance of each area from your current location."),(0,o.kt)("p",null,"Now, you have a dataset of ratings and distances for different areas. You want to use this data to find the best place to build your igloo."),(0,o.kt)("p",null,"This is where KNN comes in. It can help you find the best place to build your igloo based on the ratings and distances provided by your friends."),(0,o.kt)("p",null,"Here's how it works:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"You choose a value for K. This is the number of neighbors you want to consider when making a decision. Let's say you choose K=3."),(0,o.kt)("li",{parentName:"ol"},"You calculate the distance between each area and your current location."),(0,o.kt)("li",{parentName:"ol"},"You find the 3 areas that are closest to your current location."),(0,o.kt)("li",{parentName:"ol"},"You look at the ratings for these 3 areas, and take the average. This gives you a predicted rating for each of the 3 areas."),(0,o.kt)("li",{parentName:"ol"},"You choose the area with the highest predicted rating as the best place to build your igloo.")),(0,o.kt)("p",null,"So, in this example, KNN helped you find the best place to build your igloo based on the ratings and distances provided by your friends."),(0,o.kt)("p",null,"Of course, in real life, KNN can be used for many other things besides finding the best place to build an igloo. For example, it can be used to predict the price of a house based on its features, or to classify images based on their content."),(0,o.kt)("p",null,"But hopefully this fun example helps you understand the basic idea behind KNN!"),(0,o.kt)("hr",null),(0,o.kt)("h2",{id:"now-what-is-this-weighted-knn"},"Now what is this weighted KNN?"),(0,o.kt)("p",null,"Weighted KNN is a variation of the KNN algorithm where the contribution of each of the K nearest neighbors is weighted according to their distance from the query point. In other words, the closer a neighbor is to the query point, the more weight it is given in the final prediction."),(0,o.kt)("p",null,"In the standard KNN algorithm, all K neighbors are given equal weight in the final prediction. However, this may not always be the best approach, as some neighbors may be more relevant than others depending on their distance from the query point."),(0,o.kt)("p",null,"For example, let's say you are trying to predict the price of a house based on its features, such as the number of bedrooms, bathrooms, and square footage. In a standard KNN algorithm, the K nearest neighbors are chosen based solely on their feature values, without considering their distance from the query point. However, it's possible that some of these neighbors are located far away from the query point, and therefore may not be as relevant to the prediction."),(0,o.kt)("p",null,"In a weighted KNN algorithm, the contribution of each neighbor is weighted based on its distance from the query point. This means that neighbors that are closer to the query point are given more weight in the final prediction, while neighbors that are farther away are given less weight."),(0,o.kt)("p",null,"Using the same example of predicting house prices, this means that the K nearest neighbors are chosen based on both their feature values and their distance from the query point. The closer a neighbor is to the query point, the more weight it is given in the final prediction, as it is considered to be more relevant to the prediction."),(0,o.kt)("p",null,"Overall, weighted KNN can be a useful variation of the KNN algorithm when the distance between neighbors is an important factor in the prediction."))}d.isMDXComponent=!0}}]);