import { OnInit } from '@angular/core';
import { Donne } from './../class/Donne';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})

export class LoginComponent implements OnInit {
   
  constructor(private http: HttpClient,private router:Router) {}
  alldonne:Donne|any;
  islogin:any;
  token:any;
  ngOnInit(): void {
  
  }
  
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };
  result: string = '';

  submit (form: NgForm) {

   var user = form.value.username;
   var password=form.value.password;
    this.http
      .post<Donne|any>(
        "http://localhost:3000/api/login",
        { username: user, password: password },
        this.httpOptions
      )

      .subscribe((res) =>{
        console.log(res)
        this.alldonne=res.data
        this.islogin=res.islogin
        this.token=res.token
        

        if(this.islogin=='true'){
          console.log(this.alldonne.isAdmin)
          console.log(this.alldonne.isAdmin)
          sessionStorage.setItem('tokken',this.token)
          sessionStorage.setItem('isloggin',this.islogin)
          sessionStorage.setItem('isAdmin',this.alldonne.isAdmin)
          sessionStorage.setItem('iduser',this.alldonne.id)
          
          if(sessionStorage.getItem('url')){
            let url=sessionStorage.getItem('url')
            sessionStorage.removeItem('url')
           this.router.navigate(['/'+url])
          }
          else{
            this.router.navigate(['/'])
          }
         
         
          
        }
       
        });

    // ou
    // this.result = form.controls['username'].value;
  }

}
