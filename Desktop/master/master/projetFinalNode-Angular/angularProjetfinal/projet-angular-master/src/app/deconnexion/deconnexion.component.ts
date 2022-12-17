import { Router } from '@angular/router';
import { Component, OnInit } from '@angular/core';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-deconnexion',
  template: `
    <p>
      deconnexion works!
    </p>
  `,
  styleUrls: ['./deconnexion.component.css']
})
export class DeconnexionComponent implements OnInit {
  constructor (private router:Router){
    console.log('bonjour')
  }
  ngOnInit(): void {
    Swal.fire(
      'Deconnexion succes ',
      'Bonne fin de journée, on espére vous revoir bientot',
      'success'
    )
sessionStorage.clear()
this.router.navigate([''])

  }


}
