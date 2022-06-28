const express = require("express");
const bodyParser = require("body-parser");
const _ = require("lodash");
const { redirect } = require("express/lib/response");
const mongoose = require('mongoose');

const app = express();
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({extended: true}));
app.use(express.static("public"));

const homeContent = "Hi there to all the weebs and anime fans. Post your thoughts about your favourite anime and let the world know about your thoughts. So without wasting the time, lets get started. Compose your own posts or read the posts from the other fans";


mongoose.connect("mongodb+srv://vikrant:vikrant123@clusterpost.eybzx.mongodb.net/postDB");

const postSchema = new mongoose.Schema({
    postTime: String,
    authorName: {
        type: String,
        required: true
    },
    title: {
        type: String,
        required: true
    },
    body: {
        type: String,
        required: true
    }
});

const Post = mongoose.model("Post", postSchema);

app.get('/',(req, res)=>{

    Post.find((err, posts)=>{
        if(!err){
            res.render('home',{
                aboutYou: homeContent,
                posts: posts
            });
        }
    })
    
});

app.get('/compose', (req, res)=>{
    res.render('compose');
});

app.post('/compose', (req, res)=>{

    const options = { weekday: 'short', year: 'numeric', month: 'long', day: 'numeric' };
    const today  = new Date();
    const time = today.toLocaleDateString("en-US", options)

    const post = new Post({
        postTime: time,
        authorName: req.body.authorName,
        title: req.body.postTitle,
        body: req.body.postBody
    })
    post.save();
    res.redirect('/');

});

app.get('/about', (req, res)=>{
    res.render('about');
})

app.get('/posts/:value', (req, res)=>{
    const val1 = _.lowerCase(req.params.value);
    
    Post.find((err, posts)=>{
        for(var i=0; i<posts.length; i++){
            const val2 = _.lowerCase(posts[i].title);
    
            if(val1 === val2){
                res.render('post',{
                    postTime: posts[i].postTime,
                    authorName: posts[i].authorName,
                    title: posts[i].title,
                    body: posts[i].body
                });
            }
        }
    });

});

let port = process.env.PORT;
if (port == null || port == "") {
  port = 3000;
}

app.listen(port,function(){
    console.log("Server is running on port 3000");
});