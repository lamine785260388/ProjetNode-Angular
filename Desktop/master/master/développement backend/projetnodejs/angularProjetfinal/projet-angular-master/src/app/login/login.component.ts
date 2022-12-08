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
  alldonne:Donne;
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
      .post<Donne>(
        "http://localhost:3000/api/login",
        { username: user, password: password },
        this.httpOptions
      )

      .subscribe((data) =>{
        this.alldonne=data
        if(this.alldonne.islogin=='true'){
          console.log(this.alldonne)
          sessionStorage.setItem('tokken',this.alldonne.token)
          sessionStorage.setItem('isloggin',this.alldonne.islogin)
          if(sessionStorage.getItem('url')){
            this.router.navigate(['/'])
          }
         this.router.navigate(['/'+sessionStorage.getItem('url')])
          
        }
        else{

        }
        });

    // ou
    // this.result = form.controls['username'].value;
  }

}
