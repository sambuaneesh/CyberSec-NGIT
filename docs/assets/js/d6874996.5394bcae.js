"use strict";(self.webpackChunkdocumentation=self.webpackChunkdocumentation||[]).push([[4341],{3905:(e,t,a)=>{a.d(t,{Zo:()=>c,kt:()=>d});var r=a(7294);function n(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function l(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){n(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function o(e,t){if(null==e)return{};var a,r,n=function(e,t){if(null==e)return{};var a,r,n={},i=Object.keys(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var s=r.createContext({}),p=function(e){var t=r.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):l(l({},t),e)),a},c=function(e){var t=p(e.components);return r.createElement(s.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},h=r.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,s=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),u=p(a),h=n,d=u["".concat(s,".").concat(h)]||u[h]||m[h]||i;return a?r.createElement(d,l(l({ref:t},c),{},{components:a})):r.createElement(d,l({ref:t},c))}));function d(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,l=new Array(i);l[0]=h;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o[u]="string"==typeof e?e:n,l[1]=o;for(var p=2;p<i;p++)l[p]=a[p];return r.createElement.apply(null,l)}return r.createElement.apply(null,a)}h.displayName="MDXCreateElement"},866:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>s,contentTitle:()=>l,default:()=>m,frontMatter:()=>i,metadata:()=>o,toc:()=>p});var r=a(7462),n=(a(7294),a(3905));const i={slug:"arp-scan",title:"ARP Scan",authors:"aneesh",tags:["arp","arp-poisoning","man-in-the-middle"]},l=void 0,o={permalink:"/CyberSec-NGIT/blog/arp-scan",editUrl:"https://github.com/stealthspectre/CyberSec-NGIT/blog/arp-scan.md",source:"@site/blog/arp-scan.md",title:"ARP Scan",description:"ARP",date:"2023-03-27T18:56:21.621Z",formattedDate:"March 27, 2023",tags:[{label:"arp",permalink:"/CyberSec-NGIT/blog/tags/arp"},{label:"arp-poisoning",permalink:"/CyberSec-NGIT/blog/tags/arp-poisoning"},{label:"man-in-the-middle",permalink:"/CyberSec-NGIT/blog/tags/man-in-the-middle"}],readingTime:2.22,hasTruncateMarker:!1,authors:[{name:"Aneesh Sambu",title:"stealthspectre",url:"https://github.com/stealthspectre",imageURL:"https://github.com/stealthspectre.png",key:"aneesh"}],frontMatter:{slug:"arp-scan",title:"ARP Scan",authors:"aneesh",tags:["arp","arp-poisoning","man-in-the-middle"]},prevItem:{title:"Decision Tree Classifier",permalink:"/CyberSec-NGIT/blog/decision-tree"}},s={authorsImageUrls:[void 0]},p=[{value:"ARP",id:"arp",level:2},{value:"ARP Poisoning",id:"arp-poisoning",level:2}],c={toc:p},u="wrapper";function m(e){let{components:t,...a}=e;return(0,n.kt)(u,(0,r.Z)({},c,a,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h2",{id:"arp"},"ARP"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"Address Resolution Protocol",(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"Basic ARP identifies MAC addresses and maps them to the IP addresses"),(0,n.kt)("li",{parentName:"ul"},"the thing where it stores MAC address is known as ARP cache"))),(0,n.kt)("li",{parentName:"ul"},"arp-scan is a command-line tool that uses the ARP protocol to discover and fingerprint IP hosts on the local network"),(0,n.kt)("li",{parentName:"ul"},"arp-cache consists of associations our computer has learned about MAC addresses and IP addresses on the network",(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"initially we may get the one to the default gateway only"))),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("inlineCode",{parentName:"li"},"arp -a")," we get all the entries in the arp cache",(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"ex: ",(0,n.kt)("img",{parentName:"li",src:"https://i.imgur.com/amp52LY.png",alt:null})),(0,n.kt)("li",{parentName:"ul"},"static type means it has been feeded statically whereas dynamic type means, it had to learn about it"),(0,n.kt)("li",{parentName:"ul"},"physical address is the mapped MAC addresses"))),(0,n.kt)("li",{parentName:"ul"},"we can delete a specific entry by ",(0,n.kt)("inlineCode",{parentName:"li"},"arp -d <entry ip>")),(0,n.kt)("li",{parentName:"ul"},"but what if we want to send arp requests to an external web server",(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"we send it through the default gateway, that is in our case is the router"),(0,n.kt)("li",{parentName:"ul"},"then router sends its mac address to the requester and then the requester sends the external web IP address to router and the router handles the rest"))),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("inlineCode",{parentName:"li"},"sudo arp-scan -l")," we get all the information about hosts in the network (but with arp cache, we may not still learn all the new info about computer)"),(0,n.kt)("li",{parentName:"ul"},"but ",(0,n.kt)("inlineCode",{parentName:"li"},"-l")," is very noisy and can be easily detectable"),(0,n.kt)("li",{parentName:"ul"},"tools like netdiscover are used for stealthy scans"),(0,n.kt)("li",{parentName:"ul"},"[","[ARP Poisoning]","] is a type of MITM where hacker utilizes these ARP requests to steal info")),(0,n.kt)("h2",{id:"arp-poisoning"},"ARP Poisoning"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"let this be our initial system ",(0,n.kt)("img",{parentName:"li",src:"https://i.imgur.com/koBbVfx.png",alt:null})),(0,n.kt)("li",{parentName:"ul"},"then B turned out to be a hacker ",(0,n.kt)("img",{parentName:"li",src:"https://i.imgur.com/kaPi0rs.png",alt:null})),(0,n.kt)("li",{parentName:"ul"},"observe A's intial ARP cache for default gateway which points to the router"),(0,n.kt)("li",{parentName:"ul"},"now the hacker sends specific ARP requests to A where he changes the default gateway of A to his address, so now if A wants to communicate with the router, it sends requests to its IP in which it follows the MAC address from ARP Cache, but our hacker B has changed that IP address mapping of router in the ARP cache to his address, so all the requests will be redirected to him and first he grabs the information and then sends to the router"),(0,n.kt)("li",{parentName:"ul"},"This is called Man In The Middle Attack ",(0,n.kt)("img",{parentName:"li",src:"https://i.imgur.com/V6G5qfr.png",alt:null})),(0,n.kt)("li",{parentName:"ul"},"We again do in such a way where we want the router to send result back to us instead of A first"),(0,n.kt)("li",{parentName:"ul"},"In Kali we use Ettercap to perform this attack"),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("a",{parentName:"li",href:"https://www.youtube.com/watch?v=A7nih6SANYs"},"Procedure video")," From 3:38"),(0,n.kt)("li",{parentName:"ul"},"Using [","[DNS Cache Poisoning]","] you can make this attack more better")))}m.isMDXComponent=!0}}]);